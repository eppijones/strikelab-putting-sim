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
        
        // Update UI
        this.updateUI(data);
        
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
            
            if (data.shot.distance_cm !== undefined) {
                this.elements.shotDistance.textContent = data.shot.distance_cm.toFixed(1);
            }
        }
        
        // Performance metrics
        if (data.metrics) {
            this.elements.capFps.textContent = data.metrics.cap_fps.toFixed(1);
            this.elements.procFps.textContent = data.metrics.proc_fps.toFixed(1);
            this.elements.procLatency.textContent = data.metrics.proc_latency_ms.toFixed(1);
            this.elements.idleJitter.textContent = data.metrics.idle_stddev.toFixed(2);
        }
        
        // Ball position
        if (data.ball && data.ball.x_px !== null) {
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
        
        // Draw ball
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
        
        // Draw prediction
        if (this.prediction && this.prediction.trajectory) {
            this.drawPrediction(this.prediction);
        }
        
        requestAnimationFrame(() => this.render());
    }
    
    updateSmoothedBall(dt) {
        if (this.targetBall.x === null || this.smoothedBall.x === null) return;
        
        let predictedX = this.targetBall.x;
        let predictedY = this.targetBall.y;
        
        if (this.useVelocityPrediction && this.state && this.state.state === 'TRACKING') {
            const clampedDt = Math.min(dt, 0.1);
            predictedX = this.targetBall.x + this.ballVelocity.vx * clampedDt;
            predictedY = this.targetBall.y + this.ballVelocity.vy * clampedDt;
        }
        
        const isTracking = this.state && this.state.state === 'TRACKING';
        const factor = isTracking ? 0.5 : this.smoothingFactor;
        
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
            } else {
                this.ctx.strokeStyle = `rgba(150, 150, 150, ${alpha * 0.5})`;
            }
            
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.moveTo(p0.x, p0.y);
            this.ctx.lineTo(p1.x, p1.y);
            this.ctx.stroke();
            
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
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new PuttingSimApp();
});
