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
        
        // Smoothing/interpolation for real-time tracking
        this.smoothedBall = { x: null, y: null, radius: 12 };
        this.targetBall = { x: null, y: null, radius: 12 };
        this.ballVelocity = { vx: 0, vy: 0 };
        this.lastUpdateTime = performance.now();
        this.smoothingFactor = 0.3;  // Higher = faster response (0-1)
        this.useVelocityPrediction = true;
        
        // Calibration mode
        this.calibrating = false;
        this.calibrationPoints = [];
        
        // FPS tracking
        this.dispFrameCount = 0;
        this.lastDispFpsUpdate = performance.now();
        this.dispFps = 0;
        
        // DOM elements
        this.elements = {
            connectionStatus: document.getElementById('connection-status'),
            stateBadge: document.getElementById('state-badge'),
            speedValue: document.getElementById('speed-value'),
            directionValue: document.getElementById('direction-value'),
            framesToSpeed: document.getElementById('frames-to-speed'),
            capFps: document.getElementById('cap-fps'),
            procFps: document.getElementById('proc-fps'),
            dispFps: document.getElementById('disp-fps'),
            procLatency: document.getElementById('proc-latency'),
            trackerLane: document.getElementById('tracker-lane'),
            idleJitter: document.getElementById('idle-jitter'),
            ballX: document.getElementById('ball-x'),
            ballY: document.getElementById('ball-y'),
            calibrationStatus: document.getElementById('calibration-status'),
            calibrateBtn: document.getElementById('calibrate-btn'),
            resetBtn: document.getElementById('reset-btn'),
            calibrationModal: document.getElementById('calibration-modal'),
            calPointCount: document.getElementById('cal-point-count'),
            calWidth: document.getElementById('cal-width'),
            calHeight: document.getElementById('cal-height'),
            calCancel: document.getElementById('cal-cancel'),
            calApply: document.getElementById('cal-apply'),
            videoFeed: document.getElementById('video-feed'),
            showVideo: document.getElementById('show-video')
        };
        
        this.showVideoFeed = true;
        
        this.init();
    }
    
    init() {
        // Setup video feed
        this.setupVideoFeed();
        
        // Setup canvas
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
        
        // Setup WebSocket
        this.connectWebSocket();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Start render loop
        this.render();
    }
    
    setupVideoFeed() {
        const video = this.elements.videoFeed;
        
        // When video loads, resize canvas to match
        video.onload = () => {
            this.width = video.naturalWidth || video.width || 1280;
            this.height = video.naturalHeight || video.height || 800;
            this.resizeCanvas();
        };
        
        // Show video by default
        video.style.display = 'block';
        this.showVideoFeed = true;
        
        // Handle video toggle
        this.elements.showVideo.addEventListener('change', (e) => {
            this.showVideoFeed = e.target.checked;
            video.style.display = this.showVideoFeed ? 'block' : 'none';
            
            // Update canvas background based on video visibility
            if (!this.showVideoFeed) {
                this.canvas.style.position = 'relative';
                this.canvas.style.background = 'linear-gradient(135deg, #1a3a1a 0%, #0d1f0d 100%)';
            } else {
                this.canvas.style.position = 'absolute';
                this.canvas.style.background = 'transparent';
            }
        });
        
        // Set canvas to overlay mode initially
        this.canvas.style.position = 'absolute';
        this.canvas.style.background = 'transparent';
    }
    
    resizeCanvas() {
        const video = this.elements.videoFeed;
        
        // Get dimensions from video if available, otherwise use state
        if (video && video.naturalWidth) {
            this.width = video.naturalWidth;
            this.height = video.naturalHeight;
        }
        
        // Match canvas to video/container size
        const displayWidth = video.clientWidth || this.width;
        const displayHeight = video.clientHeight || this.height;
        
        this.scale = displayWidth / this.width;
        
        // Set canvas size to match displayed video
        this.canvas.width = displayWidth;
        this.canvas.height = displayHeight;
        
        // Scale context
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
            
            // Reconnect after delay
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
            
            // Initialize smoothed position if not set
            if (this.smoothedBall.x === null) {
                this.smoothedBall.x = this.targetBall.x;
                this.smoothedBall.y = this.targetBall.y;
                this.smoothedBall.radius = this.targetBall.radius;
            }
            
            // Store velocity for prediction-based smoothing
            if (data.velocity) {
                this.ballVelocity.vx = data.velocity.vx_px_s;
                this.ballVelocity.vy = data.velocity.vy_px_s;
            }
        }
        
        // Update trail - only during TRACKING or STOPPED (not ARMED to avoid jitter lines)
        // Use smoothed position for smoother trail
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
        
        // Clear trail and reset smoothing when transitioning to ARMED (after COOLDOWN)
        if (data.state === 'ARMED') {
            this.trail = [];
            // Reset velocity when idle
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
        
        // Update UI
        this.updateUI(data);
        
        // Record update time for interpolation
        this.lastUpdateTime = performance.now();
    }
    
    updateUI(data) {
        // State badge
        const stateClasses = {
            'ARMED': 'state-armed',
            'TRACKING': 'state-tracking',
            'STOPPED': 'state-stopped',
            'COOLDOWN': 'state-cooldown'
        };
        this.elements.stateBadge.textContent = data.state;
        this.elements.stateBadge.className = stateClasses[data.state] || 'state-armed';
        
        // Shot data
        if (data.shot) {
            this.elements.speedValue.textContent = data.shot.speed_m_s.toFixed(2);
            this.elements.directionValue.textContent = data.shot.direction_deg.toFixed(1);
            this.elements.framesToSpeed.textContent = data.shot.frames_to_speed;
        }
        
        // Performance metrics
        if (data.metrics) {
            this.elements.capFps.textContent = data.metrics.cap_fps.toFixed(1);
            this.elements.procFps.textContent = data.metrics.proc_fps.toFixed(1);
            this.elements.procLatency.textContent = data.metrics.proc_latency_ms.toFixed(1);
            this.elements.idleJitter.textContent = data.metrics.idle_stddev.toFixed(2);
        }
        
        // Tracker info
        this.elements.trackerLane.textContent = data.lane || 'IDLE';
        
        if (data.ball) {
            this.elements.ballX.textContent = data.ball.x_px?.toFixed(1) || '--';
            this.elements.ballY.textContent = data.ball.y_px?.toFixed(1) || '--';
        }
        
        // Calibration status
        this.elements.calibrationStatus.textContent = data.calibrated ? 'Calibrated' : 'Not calibrated';
        
        // Display FPS (update every 500ms)
        this.dispFrameCount++;
        const now = performance.now();
        if (now - this.lastDispFpsUpdate > 500) {
            this.dispFps = (this.dispFrameCount * 1000) / (now - this.lastDispFpsUpdate);
            this.dispFrameCount = 0;
            this.lastDispFpsUpdate = now;
            this.elements.dispFps.textContent = this.dispFps.toFixed(1);
        }
    }
    
    setupEventListeners() {
        // Calibrate button
        this.elements.calibrateBtn.addEventListener('click', () => {
            this.startCalibration();
        });
        
        // Reset button
        this.elements.resetBtn.addEventListener('click', () => {
            this.resetTracker();
        });
        
        // Canvas click for calibration - attach to container to catch all clicks
        const container = document.getElementById('canvas-container');
        container.addEventListener('click', (e) => {
            if (this.calibrating) {
                // Ignore clicks on the modal
                if (e.target.closest('.modal')) return;
                this.addCalibrationPoint(e);
            }
        });
        
        // Calibration modal
        this.elements.calCancel.addEventListener('click', () => {
            this.cancelCalibration();
        });
        
        this.elements.calApply.addEventListener('click', () => {
            this.applyCalibration();
        });
    }
    
    startCalibration() {
        this.calibrating = true;
        this.calibrationPoints = [];
        this.elements.calibrationModal.classList.remove('hidden');
        this.elements.calPointCount.textContent = '0';
        this.elements.calApply.disabled = true;
        this.canvas.style.cursor = 'crosshair';
        this.canvas.style.pointerEvents = 'auto';
        this.canvas.classList.add('interactive');
    }
    
    cancelCalibration() {
        this.calibrating = false;
        this.calibrationPoints = [];
        this.elements.calibrationModal.classList.add('hidden');
        this.canvas.style.cursor = 'default';
        this.canvas.style.pointerEvents = 'none';
        this.canvas.classList.remove('interactive');
    }
    
    addCalibrationPoint(event) {
        // Get click position relative to video feed
        const video = this.elements.videoFeed;
        const rect = video.getBoundingClientRect();
        const clickX = event.clientX - rect.left;
        const clickY = event.clientY - rect.top;
        
        // Convert to original image coordinates
        const scaleX = this.width / rect.width;
        const scaleY = this.height / rect.height;
        const x = clickX * scaleX;
        const y = clickY * scaleY;
        
        console.log(`Calibration point ${this.calibrationPoints.length + 1}: (${x.toFixed(1)}, ${y.toFixed(1)})`);
        
        this.calibrationPoints.push([x, y]);
        this.elements.calPointCount.textContent = this.calibrationPoints.length;
        
        if (this.calibrationPoints.length >= 4) {
            this.elements.calApply.disabled = false;
        }
    }
    
    async applyCalibration() {
        if (this.calibrationPoints.length < 4) return;
        
        const width = parseFloat(this.elements.calWidth.value);
        const height = parseFloat(this.elements.calHeight.value);
        
        try {
            const response = await fetch('/api/calibrate/rectangle', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    corners: this.calibrationPoints.slice(0, 4),
                    width_m: width,
                    height_m: height,
                    forward_edge: 'right'
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                console.log('Calibration successful:', result);
                this.cancelCalibration();
            } else {
                alert('Calibration failed: ' + result.error);
            }
        } catch (e) {
            console.error('Calibration error:', e);
            alert('Calibration error: ' + e.message);
        }
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
        const dt = (now - this.lastUpdateTime) / 1000;  // seconds since last WebSocket update
        
        // Update smoothed ball position with interpolation
        this.updateSmoothedBall(dt);
        
        // Clear canvas
        this.ctx.clearRect(0, 0, this.width, this.height);
        
        if (!this.showVideoFeed) {
            // Draw background only when video is hidden
            this.ctx.fillStyle = '#0d1f0d';
            this.ctx.fillRect(0, 0, this.width, this.height);
            this.drawGrid();
        }
        
        // Draw calibration points if calibrating
        if (this.calibrating) {
            this.drawCalibrationPoints();
        }
        
        // Draw trail
        this.drawTrail();
        
        // Draw ball using smoothed position
        if (this.smoothedBall.x !== null) {
            this.drawBall({
                x_px: this.smoothedBall.x,
                y_px: this.smoothedBall.y,
                radius_px: this.smoothedBall.radius
            });
        }
        
        // Draw velocity vector
        if (this.state && this.state.velocity && this.state.state === 'TRACKING') {
            this.drawVelocity(
                { x_px: this.smoothedBall.x, y_px: this.smoothedBall.y },
                this.state.velocity
            );
        }
        
        // Draw shot trajectory
        if (this.lastShot && this.lastShot.trajectory) {
            this.drawShotTrajectory(this.lastShot.trajectory);
        }
        
        // Draw prediction (when ball exits frame)
        if (this.prediction && this.prediction.trajectory) {
            this.drawPrediction(this.prediction);
        }
        
        // Request next frame
        requestAnimationFrame(() => this.render());
    }
    
    updateSmoothedBall(dt) {
        if (this.targetBall.x === null || this.smoothedBall.x === null) return;
        
        // Calculate predicted position based on velocity (for when tracking)
        let predictedX = this.targetBall.x;
        let predictedY = this.targetBall.y;
        
        // Use velocity prediction during tracking for smoother following
        if (this.useVelocityPrediction && this.state && this.state.state === 'TRACKING') {
            // Predict where the ball should be based on velocity and time since last update
            // Clamp dt to avoid huge jumps
            const clampedDt = Math.min(dt, 0.1);
            predictedX = this.targetBall.x + this.ballVelocity.vx * clampedDt;
            predictedY = this.targetBall.y + this.ballVelocity.vy * clampedDt;
        }
        
        // Exponential smoothing towards predicted/target position
        // Use faster smoothing during tracking, slower when idle
        const isTracking = this.state && this.state.state === 'TRACKING';
        const factor = isTracking ? 0.5 : this.smoothingFactor;
        
        this.smoothedBall.x += (predictedX - this.smoothedBall.x) * factor;
        this.smoothedBall.y += (predictedY - this.smoothedBall.y) * factor;
        this.smoothedBall.radius += (this.targetBall.radius - this.smoothedBall.radius) * factor;
        
        // Snap to target if very close (avoid endless micro-movements)
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
    
    drawCalibrationPoints() {
        this.ctx.fillStyle = '#e94560';
        this.ctx.strokeStyle = '#fff';
        this.ctx.lineWidth = 2;
        
        this.calibrationPoints.forEach((point, i) => {
            const [x, y] = point;
            
            // Draw point
            this.ctx.beginPath();
            this.ctx.arc(x, y, 8, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.stroke();
            
            // Draw label
            this.ctx.fillStyle = '#fff';
            this.ctx.font = '14px sans-serif';
            this.ctx.fillText((i + 1).toString(), x + 12, y + 5);
            this.ctx.fillStyle = '#e94560';
        });
        
        // Draw lines between points
        if (this.calibrationPoints.length > 1) {
            this.ctx.strokeStyle = 'rgba(233, 69, 96, 0.5)';
            this.ctx.beginPath();
            this.ctx.moveTo(this.calibrationPoints[0][0], this.calibrationPoints[0][1]);
            
            for (let i = 1; i < this.calibrationPoints.length; i++) {
                this.ctx.lineTo(this.calibrationPoints[i][0], this.calibrationPoints[i][1]);
            }
            
            if (this.calibrationPoints.length === 4) {
                this.ctx.closePath();
            }
            this.ctx.stroke();
        }
    }
    
    drawTrail() {
        if (this.trail.length < 2) return;
        
        for (let i = 1; i < this.trail.length; i++) {
            const p0 = this.trail[i - 1];
            const p1 = this.trail[i];
            
            const alpha = (i / this.trail.length) * 0.8;
            
            // Color based on state
            if (p1.state === 'TRACKING') {
                this.ctx.strokeStyle = `rgba(233, 69, 96, ${alpha})`;
            } else {
                this.ctx.strokeStyle = `rgba(150, 150, 150, ${alpha * 0.5})`;
            }
            
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.moveTo(p0.x, p0.y);
            this.ctx.lineTo(p1.x, p1.y);
            this.ctx.stroke();
            
            // Draw point
            this.ctx.fillStyle = this.ctx.strokeStyle;
            this.ctx.beginPath();
            this.ctx.arc(p1.x, p1.y, 2, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }
    
    drawBall(ball) {
        const x = ball.x_px;
        const y = ball.y_px;
        const radius = ball.radius_px || 12;
        
        // Glow effect
        const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, radius * 2);
        gradient.addColorStop(0, 'rgba(255, 255, 255, 0.3)');
        gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius * 2, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Ball
        this.ctx.fillStyle = '#fff';
        this.ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.stroke();
    }
    
    drawVelocity(ball, velocity) {
        if (!ball || !velocity) return;
        
        const x = ball.x_px;
        const y = ball.y_px;
        
        // Scale velocity for display (pixels per second -> display length)
        const scale = 0.05;
        const vx = velocity.vx_px_s * scale;
        const vy = velocity.vy_px_s * scale;
        
        const length = Math.sqrt(vx * vx + vy * vy);
        if (length < 5) return;
        
        // Arrow line
        this.ctx.strokeStyle = '#4ade80';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        this.ctx.moveTo(x, y);
        this.ctx.lineTo(x + vx, y + vy);
        this.ctx.stroke();
        
        // Arrow head
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
    
    drawShotTrajectory(trajectory) {
        if (!trajectory || trajectory.length < 2) return;
        
        this.ctx.strokeStyle = 'rgba(251, 191, 36, 0.3)';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        
        this.ctx.beginPath();
        this.ctx.moveTo(trajectory[0][0], trajectory[0][1]);
        
        for (let i = 1; i < trajectory.length; i++) {
            this.ctx.lineTo(trajectory[i][0], trajectory[i][1]);
        }
        
        this.ctx.stroke();
        this.ctx.setLineDash([]);
    }
    
    drawPrediction(prediction) {
        const trajectory = prediction.trajectory;
        if (!trajectory || trajectory.length < 2) return;
        
        // Draw predicted path as dashed line
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
        
        // Draw final predicted position
        if (prediction.final_position) {
            const [fx, fy] = prediction.final_position;
            
            // Only draw if within or near frame bounds
            if (fx > -100 && fx < this.width + 500 && fy > -100 && fy < this.height + 100) {
                // Target marker
                this.ctx.strokeStyle = 'rgba(251, 191, 36, 0.8)';
                this.ctx.lineWidth = 2;
                
                // Crosshair
                const size = 15;
                this.ctx.beginPath();
                this.ctx.moveTo(fx - size, fy);
                this.ctx.lineTo(fx + size, fy);
                this.ctx.moveTo(fx, fy - size);
                this.ctx.lineTo(fx, fy + size);
                this.ctx.stroke();
                
                // Circle
                this.ctx.beginPath();
                this.ctx.arc(fx, fy, 10, 0, Math.PI * 2);
                this.ctx.stroke();
                
                // Label
                this.ctx.fillStyle = 'rgba(251, 191, 36, 0.9)';
                this.ctx.font = '12px sans-serif';
                this.ctx.fillText(
                    `${prediction.final_time_s}s`,
                    fx + 15,
                    fy - 5
                );
            }
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new PuttingSimApp();
});
