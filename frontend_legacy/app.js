/**
 * StrikeLab Putting Sim - Frontend Application
 * 
 * Canvas rendering + WebSocket client for real-time ball tracking visualization.
 */

class PuttingSimApp {
    constructor() {
        // Canvas setup
        this.canvas = document.getElementById('ball-canvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Resolution (will be updated from server)
        this.width = 1280;
        this.height = 800;
        this.scale = 1;
        
        // State
        this.connected = false;
        this.state = null;
        this.trail = [];
        this.maxTrailLength = 50;
        this.lastShot = null;
        this.prediction = null;
        this.virtualBall = null;  // Virtual ball state for off-frame rolling
        this.exitState = null;    // State when ball exited frame
        
        // Smoothing/interpolation for real-time tracking
        this.smoothedBall = { x: null, y: null, radius: 12 };
        this.targetBall = { x: null, y: null, radius: 12 };
        this.ballVelocity = { vx: 0, vy: 0 };
        this.lastUpdateTime = performance.now();
        this.smoothingFactor = 0.3;
        this.useVelocityPrediction = true;
        
        // DOM elements
        this.elements = {
            connectionStatus: document.getElementById('connection-status'),
            stateBadge: document.getElementById('state-badge'),
            speedValue: document.getElementById('speed-value'),
            directionValue: document.getElementById('direction-value'),
            shotDistance: document.getElementById('shot-distance'),
            capFps: document.getElementById('cap-fps'),
            procFps: document.getElementById('proc-fps'),
            procLatency: document.getElementById('proc-latency'),
            calibrationStatus: document.getElementById('calibration-status'),
            ballPosition: document.getElementById('ball-position'),
            idleJitter: document.getElementById('idle-jitter'),
            resetBtn: document.getElementById('reset-btn'),
            videoFeed: document.getElementById('video-feed'),
            showVideo: document.getElementById('show-video')
        };
        
        this.showVideoFeed = true;
        
        this.init();
    }
    
    init() {
        this.setupVideoFeed();
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
        this.connectWebSocket();
        this.setupEventListeners();
        this.render();
    }
    
    setupVideoFeed() {
        const video = this.elements.videoFeed;
        
        video.onload = () => {
            this.width = video.naturalWidth || video.width || 1280;
            this.height = video.naturalHeight || video.height || 800;
            this.resizeCanvas();
        };
        
        video.style.display = 'block';
        this.showVideoFeed = true;
        
        this.elements.showVideo.addEventListener('change', (e) => {
            this.showVideoFeed = e.target.checked;
            video.style.display = this.showVideoFeed ? 'block' : 'none';
            
            if (!this.showVideoFeed) {
                this.canvas.style.position = 'relative';
                this.canvas.style.background = 'linear-gradient(135deg, #1a3a1a 0%, #0d1f0d 100%)';
            } else {
                this.canvas.style.position = 'absolute';
                this.canvas.style.background = 'transparent';
            }
        });
        
        this.canvas.style.position = 'absolute';
        this.canvas.style.background = 'transparent';
    }
    
    resizeCanvas() {
        const video = this.elements.videoFeed;
        
        if (video && video.naturalWidth) {
            this.width = video.naturalWidth;
            this.height = video.naturalHeight;
        }
        
        const displayWidth = video.clientWidth || this.width;
        const displayHeight = video.clientHeight || this.height;
        
        this.scale = displayWidth / this.width;
        
        this.canvas.width = displayWidth;
        this.canvas.height = displayHeight;
        
        this.ctx.setTransform(this.scale, 0, 0, this.scale, 0, 0);
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        console.log('Connecting to WebSocket:', wsUrl);
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.connected = true;
            this.elements.connectionStatus.textContent = 'Connected';
            this.elements.connectionStatus.className = 'status connected';
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.connected = false;
            this.elements.connectionStatus.textContent = 'Disconnected';
            this.elements.connectionStatus.className = 'status disconnected';
            setTimeout(() => this.connectWebSocket(), 2000);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'ping') {
                    this.ws.send(JSON.stringify({ type: 'pong' }));
                    return;
                }
                this.handleState(data);
            } catch (e) {
                console.error('Failed to parse message:', e);
            }
        };
    }
    
    handleState(data) {
        this.state = data;
        
        // Update resolution if provided
        if (data.resolution) {
            const [w, h] = data.resolution;
            if (w !== this.width || h !== this.height) {
                this.width = w;
                this.height = h;
                this.resizeCanvas();
            }
        }
        
        // Update target ball position for smoothing
        if (data.ball && data.ball.x_px !== null) {
            this.targetBall.x = data.ball.x_px;
            this.targetBall.y = data.ball.y_px;
            this.targetBall.radius = data.ball.radius_px || 12;
            
            if (this.smoothedBall.x === null) {
                this.smoothedBall.x = this.targetBall.x;
                this.smoothedBall.y = this.targetBall.y;
                this.smoothedBall.radius = this.targetBall.radius;
            }
            
            if (data.velocity) {
                this.ballVelocity.vx = data.velocity.vx_px_s;
                this.ballVelocity.vy = data.velocity.vy_px_s;
            }
        }
        
        // Update trail during TRACKING or STOPPED
        if (data.ball && data.ball.x_px !== null && (data.state === 'TRACKING' || data.state === 'STOPPED')) {
            this.trail.push({
                x: data.ball.x_px,
                y: data.ball.y_px,
                state: data.state
            });
            
            if (this.trail.length > this.maxTrailLength) {
                this.trail.shift();
            }
        }
        
        // Clear trail when transitioning to ARMED
        if (data.state === 'ARMED') {
            this.trail = [];
            this.ballVelocity.vx = 0;
            this.ballVelocity.vy = 0;
        }
        
        // Store shot result
        if (data.shot) {
            this.lastShot = data.shot;
        }
        
        // Store prediction
        if (data.prediction) {
            this.prediction = data.prediction;
        } else if (data.state === 'ARMED') {
            this.prediction = null;
        }
        
        // Store virtual ball state
        if (data.virtual_ball) {
            this.virtualBall = data.virtual_ball;
            // Update target position to virtual ball position for smooth rendering
            this.targetBall.x = data.virtual_ball.x;
            this.targetBall.y = data.virtual_ball.y;
            
            // Add virtual position to trail
            this.trail.push({
                x: data.virtual_ball.x,
                y: data.virtual_ball.y,
                state: 'VIRTUAL_ROLLING'
            });
            if (this.trail.length > this.maxTrailLength * 2) {  // Allow longer trail for virtual
                this.trail.shift();
            }
        } else if (data.state === 'ARMED') {
            this.virtualBall = null;
        }
        
        // Store exit state
        if (data.exit_state) {
            this.exitState = data.exit_state;
        } else if (data.state === 'ARMED') {
            this.exitState = null;
        }
        
        // Update UI
        this.updateUI(data);
        
        this.lastUpdateTime = performance.now();
    }
    
    updateUI(data) {
        // State badge
        const stateClasses = {
            'ARMED': 'state-armed',
            'TRACKING': 'state-tracking',
            'VIRTUAL_ROLLING': 'state-virtual',
            'STOPPED': 'state-stopped',
            'COOLDOWN': 'state-cooldown'
        };
        
        // Show friendly name for virtual rolling
        const stateNames = {
            'ARMED': 'ARMED',
            'TRACKING': 'TRACKING',
            'VIRTUAL_ROLLING': 'ROLLING...',
            'STOPPED': 'STOPPED',
            'COOLDOWN': 'COOLDOWN'
        };
        
        this.elements.stateBadge.textContent = stateNames[data.state] || data.state;
        this.elements.stateBadge.className = stateClasses[data.state] || 'state-armed';
        
        // Shot data - show total distance (physical + virtual)
        if (data.shot) {
            this.elements.speedValue.textContent = data.shot.speed_m_s.toFixed(2);
            this.elements.directionValue.textContent = data.shot.direction_deg.toFixed(1);
            
            // Show total distance (including virtual)
            if (data.shot.distance_cm !== undefined) {
                const distCm = data.shot.distance_cm;
                const distM = data.shot.distance_m;
                
                // Show in meters if >= 1m, otherwise cm
                if (distM >= 1) {
                    this.elements.shotDistance.textContent = distM.toFixed(2) + 'm';
                } else {
                    this.elements.shotDistance.textContent = distCm.toFixed(1);
                }
            }
        }
        
        // Live distance during virtual rolling
        if (data.state === 'VIRTUAL_ROLLING' && data.virtual_ball) {
            const vb = data.virtual_ball;
            // Get physical distance from exit state
            let physicalDistCm = 0;
            let virtualDistCm = vb.distance_cm || 0;
            
            // Add physical distance if we have exit state with trajectory
            if (data.exit_state && data.exit_state.trajectory_before_exit) {
                const traj = data.exit_state.trajectory_before_exit;
                if (traj.length >= 2) {
                    const start = traj[0];
                    const end = traj[traj.length - 1];
                    const physicalPx = Math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2);
                    physicalDistCm = (physicalPx / data.pixels_per_meter) * 100;
                }
            }
            
            const totalDistCm = physicalDistCm + virtualDistCm;
            const totalDistM = totalDistCm / 100;
            
            // Update distance display with live total - format nicely
            if (totalDistM >= 1) {
                this.elements.shotDistance.textContent = totalDistM.toFixed(2) + 'm';
            } else {
                this.elements.shotDistance.textContent = totalDistCm.toFixed(0) + '';
            }
            
            // Update speed display with virtual ball speed
            this.elements.speedValue.textContent = (vb.speed_m_s || 0).toFixed(2);
            
            // Log to console for debugging
            console.log(`Virtual rolling: physical=${physicalDistCm.toFixed(0)}cm + virtual=${virtualDistCm.toFixed(0)}cm = ${totalDistCm.toFixed(0)}cm total`);
        }
        
        // Performance metrics
        if (data.metrics) {
            this.elements.capFps.textContent = data.metrics.cap_fps.toFixed(1);
            this.elements.procFps.textContent = data.metrics.proc_fps.toFixed(1);
            this.elements.procLatency.textContent = data.metrics.proc_latency_ms.toFixed(1);
            this.elements.idleJitter.textContent = data.metrics.idle_stddev.toFixed(2);
        }
        
        // Ball position - show virtual position during virtual rolling
        if (data.state === 'VIRTUAL_ROLLING' && data.virtual_ball) {
            this.elements.ballPosition.textContent = `(${data.virtual_ball.x.toFixed(0)}, ${data.virtual_ball.y.toFixed(0)}) [V]`;
        } else if (data.ball && data.ball.x_px !== null) {
            this.elements.ballPosition.textContent = `(${data.ball.x_px.toFixed(0)}, ${data.ball.y_px.toFixed(0)})`;
        }
        
        // Calibration status
        if (data.auto_calibrated) {
            this.elements.calibrationStatus.textContent = 'Ready';
            this.elements.calibrationStatus.classList.add('ready');
        } else if (data.calibrated) {
            this.elements.calibrationStatus.textContent = 'Ready';
            this.elements.calibrationStatus.classList.add('ready');
        } else {
            this.elements.calibrationStatus.textContent = 'Calibrating...';
            this.elements.calibrationStatus.classList.remove('ready');
        }
    }
    
    setupEventListeners() {
        // Reset button
        this.elements.resetBtn.addEventListener('click', () => {
            this.resetTracker();
        });
    }
    
    async resetTracker() {
        try {
            await fetch('/api/tracker/reset', { method: 'POST' });
            this.trail = [];
            this.lastShot = null;
        } catch (e) {
            console.error('Reset error:', e);
        }
    }
    
    render() {
        const now = performance.now();
        const dt = (now - this.lastUpdateTime) / 1000;
        
        this.updateSmoothedBall(dt);
        
        // Clear canvas
        this.ctx.clearRect(0, 0, this.width, this.height);
        
        if (!this.showVideoFeed) {
            this.ctx.fillStyle = '#0d1f0d';
            this.ctx.fillRect(0, 0, this.width, this.height);
            this.drawGrid();
        }
        
        // Draw trail
        this.drawTrail();
        
        // Draw exit point marker if ball has exited
        if (this.exitState) {
            this.drawExitMarker(this.exitState);
        }
        
        // Draw ball - use different style for virtual rolling
        const isVirtualRolling = this.state && this.state.state === 'VIRTUAL_ROLLING';
        if (this.smoothedBall.x !== null) {
            if (isVirtualRolling) {
                this.drawVirtualBall({
                    x_px: this.smoothedBall.x,
                    y_px: this.smoothedBall.y,
                    radius_px: this.smoothedBall.radius
                });
            } else {
                this.drawBall({
                    x_px: this.smoothedBall.x,
                    y_px: this.smoothedBall.y,
                    radius_px: this.smoothedBall.radius
                });
            }
        }
        
        // Draw velocity vector
        if (this.state && this.state.velocity && this.state.state === 'TRACKING') {
            this.drawVelocity(
                { x_px: this.smoothedBall.x, y_px: this.smoothedBall.y },
                this.state.velocity
            );
        }
        
        // Draw virtual ball velocity and info
        if (isVirtualRolling && this.virtualBall) {
            this.drawVirtualBallInfo(this.virtualBall);
        }
        
        // Draw prediction (when not in virtual rolling - virtual rolling handles its own display)
        if (this.prediction && this.prediction.trajectory && !isVirtualRolling) {
            this.drawPrediction(this.prediction);
        }
        
        // Draw final position marker during virtual rolling
        if (isVirtualRolling && this.virtualBall && this.virtualBall.final_position) {
            this.drawFinalPosition(this.virtualBall.final_position);
        }
        
        requestAnimationFrame(() => this.render());
    }
    
    updateSmoothedBall(dt) {
        if (this.targetBall.x === null || this.smoothedBall.x === null) return;
        
        let predictedX = this.targetBall.x;
        let predictedY = this.targetBall.y;
        
        const isTracking = this.state && this.state.state === 'TRACKING';
        const isVirtualRolling = this.state && this.state.state === 'VIRTUAL_ROLLING';
        
        if (this.useVelocityPrediction && (isTracking || isVirtualRolling)) {
            const clampedDt = Math.min(dt, 0.1);
            
            // Use virtual ball velocity during virtual rolling
            if (isVirtualRolling && this.virtualBall) {
                predictedX = this.targetBall.x + this.virtualBall.vx * clampedDt;
                predictedY = this.targetBall.y + this.virtualBall.vy * clampedDt;
            } else {
                predictedX = this.targetBall.x + this.ballVelocity.vx * clampedDt;
                predictedY = this.targetBall.y + this.ballVelocity.vy * clampedDt;
            }
        }
        
        // Faster smoothing during tracking/virtual rolling for responsiveness
        const factor = (isTracking || isVirtualRolling) ? 0.5 : this.smoothingFactor;
        
        this.smoothedBall.x += (predictedX - this.smoothedBall.x) * factor;
        this.smoothedBall.y += (predictedY - this.smoothedBall.y) * factor;
        this.smoothedBall.radius += (this.targetBall.radius - this.smoothedBall.radius) * factor;
        
        const dx = predictedX - this.smoothedBall.x;
        const dy = predictedY - this.smoothedBall.y;
        if (dx * dx + dy * dy < 0.5) {
            this.smoothedBall.x = predictedX;
            this.smoothedBall.y = predictedY;
        }
    }
    
    drawGrid() {
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        this.ctx.lineWidth = 1;
        
        const gridSize = 100;
        
        for (let x = 0; x <= this.width; x += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.height);
            this.ctx.stroke();
        }
        
        for (let y = 0; y <= this.height; y += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.width, y);
            this.ctx.stroke();
        }
    }
    
    drawTrail() {
        if (this.trail.length < 2) return;
        
        for (let i = 1; i < this.trail.length; i++) {
            const p0 = this.trail[i - 1];
            const p1 = this.trail[i];
            
            const alpha = (i / this.trail.length) * 0.8;
            
            if (p1.state === 'TRACKING') {
                this.ctx.strokeStyle = `rgba(233, 69, 96, ${alpha})`;
            } else if (p1.state === 'VIRTUAL_ROLLING') {
                // Cyan/blue trail for virtual rolling
                this.ctx.strokeStyle = `rgba(100, 200, 255, ${alpha})`;
            } else {
                this.ctx.strokeStyle = `rgba(150, 150, 150, ${alpha * 0.5})`;
            }
            
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.moveTo(p0.x, p0.y);
            this.ctx.lineTo(p1.x, p1.y);
            this.ctx.stroke();
            
            // Draw dots - smaller for virtual rolling
            this.ctx.fillStyle = this.ctx.strokeStyle;
            this.ctx.beginPath();
            const dotSize = p1.state === 'VIRTUAL_ROLLING' ? 1.5 : 2;
            this.ctx.arc(p1.x, p1.y, dotSize, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }
    
    drawBall(ball) {
        const x = ball.x_px;
        const y = ball.y_px;
        const detectedRadius = ball.radius_px || 12;
        
        // Apply overlay scale for display ONLY - does not affect tracking
        // This corrects for the ~13% underdraw of the detected radius
        const overlayScale = this.state?.overlay_radius_scale || 1.15;
        const displayRadius = detectedRadius * overlayScale;
        
        // Glow effect
        const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, displayRadius * 2);
        gradient.addColorStop(0, 'rgba(255, 255, 255, 0.3)');
        gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(x, y, displayRadius * 2, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Ball with scaled radius
        this.ctx.fillStyle = '#fff';
        this.ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        this.ctx.arc(x, y, displayRadius, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.stroke();
    }
    
    drawVelocity(ball, velocity) {
        if (!ball || !velocity) return;
        
        const x = ball.x_px;
        const y = ball.y_px;
        
        const scale = 0.05;
        const vx = velocity.vx_px_s * scale;
        const vy = velocity.vy_px_s * scale;
        
        const length = Math.sqrt(vx * vx + vy * vy);
        if (length < 5) return;
        
        this.ctx.strokeStyle = '#4ade80';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        this.ctx.moveTo(x, y);
        this.ctx.lineTo(x + vx, y + vy);
        this.ctx.stroke();
        
        const angle = Math.atan2(vy, vx);
        const headLength = 10;
        
        this.ctx.fillStyle = '#4ade80';
        this.ctx.beginPath();
        this.ctx.moveTo(x + vx, y + vy);
        this.ctx.lineTo(
            x + vx - headLength * Math.cos(angle - Math.PI / 6),
            y + vy - headLength * Math.sin(angle - Math.PI / 6)
        );
        this.ctx.lineTo(
            x + vx - headLength * Math.cos(angle + Math.PI / 6),
            y + vy - headLength * Math.sin(angle + Math.PI / 6)
        );
        this.ctx.closePath();
        this.ctx.fill();
    }
    
    drawPrediction(prediction) {
        const trajectory = prediction.trajectory;
        if (!trajectory || trajectory.length < 2) return;
        
        this.ctx.strokeStyle = 'rgba(251, 191, 36, 0.6)';
        this.ctx.lineWidth = 3;
        this.ctx.setLineDash([10, 5]);
        
        this.ctx.beginPath();
        this.ctx.moveTo(trajectory[0][0], trajectory[0][1]);
        
        for (let i = 1; i < trajectory.length; i++) {
            this.ctx.lineTo(trajectory[i][0], trajectory[i][1]);
        }
        
        this.ctx.stroke();
        this.ctx.setLineDash([]);
        
        if (prediction.final_position) {
            const [fx, fy] = prediction.final_position;
            
            if (fx > -100 && fx < this.width + 500 && fy > -100 && fy < this.height + 100) {
                this.ctx.strokeStyle = 'rgba(251, 191, 36, 0.8)';
                this.ctx.lineWidth = 2;
                
                const size = 15;
                this.ctx.beginPath();
                this.ctx.moveTo(fx - size, fy);
                this.ctx.lineTo(fx + size, fy);
                this.ctx.moveTo(fx, fy - size);
                this.ctx.lineTo(fx, fy + size);
                this.ctx.stroke();
                
                this.ctx.beginPath();
                this.ctx.arc(fx, fy, 10, 0, Math.PI * 2);
                this.ctx.stroke();
            }
        }
    }
    
    drawVirtualBall(ball) {
        const x = ball.x_px;
        const y = ball.y_px;
        const detectedRadius = ball.radius_px || 12;
        
        // Apply overlay scale for display ONLY - does not affect tracking
        const overlayScale = this.state?.overlay_radius_scale || 1.15;
        const displayRadius = detectedRadius * overlayScale;
        
        // Pulsing glow effect for virtual ball
        const time = performance.now() / 500;
        const pulseIntensity = 0.3 + 0.2 * Math.sin(time);
        
        // Larger, more dramatic glow
        const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, displayRadius * 3);
        gradient.addColorStop(0, `rgba(100, 200, 255, ${pulseIntensity})`);
        gradient.addColorStop(0.5, `rgba(100, 200, 255, ${pulseIntensity * 0.5})`);
        gradient.addColorStop(1, 'rgba(100, 200, 255, 0)');
        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(x, y, displayRadius * 3, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Virtual ball - semi-transparent with cyan tint
        this.ctx.fillStyle = 'rgba(200, 230, 255, 0.8)';
        this.ctx.strokeStyle = 'rgba(100, 200, 255, 0.9)';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.arc(x, y, displayRadius, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.stroke();
        
        // Inner highlight
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
        this.ctx.beginPath();
        this.ctx.arc(x - displayRadius * 0.3, y - displayRadius * 0.3, displayRadius * 0.3, 0, Math.PI * 2);
        this.ctx.fill();
    }
    
    drawExitMarker(exitState) {
        if (!exitState || !exitState.position) return;
        
        const [x, y] = exitState.position;
        
        // Draw a subtle marker where ball exited frame
        this.ctx.strokeStyle = 'rgba(100, 200, 255, 0.5)';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        
        // Small cross
        const size = 10;
        this.ctx.beginPath();
        this.ctx.moveTo(x - size, y);
        this.ctx.lineTo(x + size, y);
        this.ctx.moveTo(x, y - size);
        this.ctx.lineTo(x, y + size);
        this.ctx.stroke();
        
        // Circle around exit point
        this.ctx.beginPath();
        this.ctx.arc(x, y, size * 1.5, 0, Math.PI * 2);
        this.ctx.stroke();
        
        this.ctx.setLineDash([]);
    }
    
    drawFinalPosition(finalPosition) {
        if (!finalPosition) return;
        
        const [fx, fy] = finalPosition;
        
        // Only draw if reasonably within extended view
        if (fx < -200 || fx > this.width + 2000 || fy < -200 || fy > this.height + 200) return;
        
        // Pulsing target marker
        const time = performance.now() / 300;
        const pulse = 0.6 + 0.4 * Math.sin(time);
        
        this.ctx.strokeStyle = `rgba(76, 217, 100, ${pulse})`;
        this.ctx.lineWidth = 3;
        
        // Target circle
        this.ctx.beginPath();
        this.ctx.arc(fx, fy, 15, 0, Math.PI * 2);
        this.ctx.stroke();
        
        // Crosshair
        const size = 25;
        this.ctx.beginPath();
        this.ctx.moveTo(fx - size, fy);
        this.ctx.lineTo(fx + size, fy);
        this.ctx.moveTo(fx, fy - size);
        this.ctx.lineTo(fx, fy + size);
        this.ctx.stroke();
        
        // Inner dot
        this.ctx.fillStyle = `rgba(76, 217, 100, ${pulse})`;
        this.ctx.beginPath();
        this.ctx.arc(fx, fy, 4, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Draw "STOP" label
        this.ctx.fillStyle = `rgba(76, 217, 100, ${pulse})`;
        this.ctx.font = 'bold 12px monospace';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('STOP', fx, fy - 25);
    }
    
    drawVirtualBallInfo(virtualBall) {
        if (!virtualBall) return;
        
        // Draw velocity vector from virtual ball
        if (virtualBall.speed_px_s > 20) {
            const x = virtualBall.x;
            const y = virtualBall.y;
            const scale = 0.05;
            const vx = virtualBall.vx * scale;
            const vy = virtualBall.vy * scale;
            
            this.ctx.strokeStyle = 'rgba(100, 200, 255, 0.8)';
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.moveTo(x, y);
            this.ctx.lineTo(x + vx, y + vy);
            this.ctx.stroke();
            
            // Arrowhead
            const angle = Math.atan2(vy, vx);
            const headLength = 8;
            this.ctx.fillStyle = 'rgba(100, 200, 255, 0.8)';
            this.ctx.beginPath();
            this.ctx.moveTo(x + vx, y + vy);
            this.ctx.lineTo(
                x + vx - headLength * Math.cos(angle - Math.PI / 6),
                y + vy - headLength * Math.sin(angle - Math.PI / 6)
            );
            this.ctx.lineTo(
                x + vx - headLength * Math.cos(angle + Math.PI / 6),
                y + vy - headLength * Math.sin(angle + Math.PI / 6)
            );
            this.ctx.closePath();
            this.ctx.fill();
        }
        
        // Draw distance info near the ball
        const x = Math.min(virtualBall.x, this.width - 100);
        const y = Math.max(virtualBall.y - 40, 30);
        
        this.ctx.fillStyle = 'rgba(100, 200, 255, 0.9)';
        this.ctx.font = 'bold 14px monospace';
        this.ctx.textAlign = 'left';
        
        // Distance traveled
        const distCm = virtualBall.distance_cm || 0;
        const distM = virtualBall.distance_m || 0;
        
        if (distM >= 1) {
            this.ctx.fillText(`${distM.toFixed(2)}m`, x + 20, y);
        } else {
            this.ctx.fillText(`${distCm.toFixed(0)}cm`, x + 20, y);
        }
        
        // Speed
        const speedMs = virtualBall.speed_m_s || 0;
        this.ctx.fillText(`${speedMs.toFixed(2)} m/s`, x + 20, y + 16);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new PuttingSimApp();
});
