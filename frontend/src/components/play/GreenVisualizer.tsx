import React, { useRef, useEffect, useState } from 'react';
import { usePuttingState } from '../../contexts/WebSocketContext';

export const GreenVisualizer: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const { ballPosition, gameState, pixelsPerMeter, lastJsonMessage } = usePuttingState();
  
  // Configuration
  const [holeDistanceM, setHoleDistanceM] = useState(3.0); // 3 meters default
  // Remove unused viewHeightM state if it's static, or use it
  const viewHeightM = 4.0; // Show 4m total
  
  // Camera resolution for coordinate mapping
  const camWidth = lastJsonMessage?.resolution[0] || 1280;
  const camHeight = lastJsonMessage?.resolution[1] || 800;

  // Render Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle resizing
    const updateSize = () => {
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
    };
    updateSize();
    window.addEventListener('resize', updateSize);

    // --- Drawing Helper Functions ---

    // Scale: Screen pixels per Physical meter
    // We want 'viewHeightM' to fit in 'canvas.height'
    const scale = canvas.height / viewHeightM;

    // Coordinate Transform: 
    // Assumes Camera X+ is Forward (Putt Direction)
    // Assumes Camera Y is Horizontal (Left/Right)
    // Screen X: Centered on Camera Y
    // Screen Y: Bottom - Camera X
    
    const transformX = (camY: number) => {
      // Center camY (0..800) to screen center
      const camCenterY = camHeight / 2;
      const offsetM = (camCenterY - camY) / pixelsPerMeter;
      return (canvas.width / 2) + (offsetM * scale);
    };

    const transformY = (camX: number) => {
      // camX starts at 0 (bottom of screen/tee) and goes to camWidth (top of frame)
      // Then virtual distance adds to that.
      const distM = camX / pixelsPerMeter;
      return canvas.height - (distM * scale) - (canvas.height * 0.1); // 10% padding at bottom
    };
    
    // Virtual Ball Transform (already in meters if we look at tracker logic, but let's check)
    // VirtualBall state in backend/tracker.py: x, y are "virtual coordinates".
    // Usually Tracker continues the coordinate system of the camera.
    // So Virtual X is > camWidth.
    
    // Draw Frame
    const render = () => {
      // 1. Clear
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // 2. Draw Grid / Green Texture
      drawGrid(ctx, canvas.width, canvas.height, scale);

      // 3. Draw Physical/Virtual Line (Camera limit)
      const camLimitY = transformY(camWidth);
      ctx.beginPath();
      ctx.moveTo(0, camLimitY);
      ctx.lineTo(canvas.width, camLimitY);
      ctx.strokeStyle = 'rgba(6, 182, 212, 0.3)'; // sl-cyan low opacity
      ctx.setLineDash([5, 5]);
      ctx.stroke();
      ctx.setLineDash([]);
      
      // Label "Camera View"
      ctx.fillStyle = 'rgba(6, 182, 212, 0.5)';
      ctx.font = '10px monospace';
      ctx.fillText('CAMERA LIMIT', 10, camLimitY - 5);

      // 4. Draw Hole
      const holeY = transformY(holeDistanceM * pixelsPerMeter);
      const holeX = canvas.width / 2; // Assume straight putt for now
      const holeRadius = (0.108 / 2) * scale; // Standard cup 10.8cm
      
      ctx.beginPath();
      ctx.arc(holeX, holeY, holeRadius, 0, Math.PI * 2);
      ctx.fillStyle = '#1e293b'; // Slate 800 (hole depth)
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.1)';
      ctx.stroke();

      // Flag/Pin
      ctx.beginPath();
      ctx.moveTo(holeX, holeY);
      ctx.lineTo(holeX, holeY - 40);
      ctx.strokeStyle = '#f59e0b'; // sl-accent
      ctx.stroke();
      
      // 5. Draw Ball
      if (ballPosition) {
        // Check if ball is "virtual" (from tracker state) or "physical"
        // The WebSocketContext normalizes this into `ballPosition` {x, y}
        // which are in "Camera Pixel Space" (even if virtual, they extend beyond camWidth)
        
        const bx = transformX(ballPosition.y); // Note swap: CamY -> ScreenX
        const by = transformY(ballPosition.x); // Note swap: CamX -> ScreenY
        
        const ballRadiusM = 0.04267 / 2; // 42.67mm
        const ballRadiusScreen = ballRadiusM * scale;

        // Shadow
        ctx.beginPath();
        ctx.ellipse(bx + 2, by + 2, ballRadiusScreen, ballRadiusScreen * 0.8, 0, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0,0,0,0.3)';
        ctx.fill();

        // Ball Body
        ctx.beginPath();
        ctx.arc(bx, by, ballRadiusScreen, 0, Math.PI * 2);
        ctx.fillStyle = 'white';
        ctx.fill();
        
        // Ball Glow (if moving)
        if (gameState === 'TRACKING' || gameState === 'VIRTUAL_ROLLING') {
           const gradient = ctx.createRadialGradient(bx, by, ballRadiusScreen, bx, by, ballRadiusScreen * 3);
           gradient.addColorStop(0, 'rgba(34, 197, 94, 0.4)'); // sl-green
           gradient.addColorStop(1, 'rgba(34, 197, 94, 0)');
           ctx.fillStyle = gradient;
           ctx.beginPath();
           ctx.arc(bx, by, ballRadiusScreen * 3, 0, Math.PI * 2);
           ctx.fill();
        }
      } else if (gameState === 'ARMED') {
         // Draw ghost ball at origin (Ready State)
         const bx = transformX(400); // Center Y -> ScreenX (assuming 800w center)
         const by = transformY(0);   // Bottom X -> ScreenY (Tee)
         
         const ballRadiusM = 0.04267 / 2;
         const ballRadiusScreen = ballRadiusM * scale;
         
         ctx.beginPath();
         ctx.arc(bx, by, ballRadiusScreen, 0, Math.PI * 2);
         ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
         ctx.fill();
         ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
         ctx.setLineDash([2, 2]);
         ctx.lineWidth = 1;
         ctx.stroke();
         ctx.setLineDash([]);
      }

      requestAnimationFrame(render);
    };

    const animationId = requestAnimationFrame(render);
    return () => {
      window.removeEventListener('resize', updateSize);
      cancelAnimationFrame(animationId);
    };
  }, [ballPosition, gameState, pixelsPerMeter, holeDistanceM, viewHeightM, camWidth, camHeight]);

  const drawGrid = (ctx: CanvasRenderingContext2D, w: number, h: number, scale: number) => {
    // Vertical lines (1m apart)
    const centerX = w / 2;
    const lines = 3; // 3m either side
    
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.lineWidth = 1;

    for (let i = -lines; i <= lines; i++) {
      const x = centerX + (i * scale); // 1m = scale pixels
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }

    // Horizontal lines (1m apart)
    // Start from bottom offset
    const bottomY = h - (h * 0.1); // 10% padding
    const metersTotal = h / scale;
    
    for (let i = 0; i <= metersTotal; i++) {
      const y = bottomY - (i * scale);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
      
      // Distance Marker
      if (i > 0) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.font = '10px sans-serif';
        ctx.fillText(`${i}m`, w - 30, y - 5);
      }
    }
  };

  return (
    <div ref={containerRef} className="w-full h-full relative bg-sl-dark overflow-hidden">
      <canvas ref={canvasRef} className="block" />
      
      {/* Distance Control Overlay (Simple) */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-sl-panel/80 rounded-full px-4 py-2 border border-white/10 backdrop-blur text-sm">
        <span className="text-gray-400">Hole Distance:</span>
        <button 
          onClick={() => setHoleDistanceM(Math.max(1, holeDistanceM - 0.5))}
          className="hover:text-sl-green font-bold px-2"
        >-</button>
        <span className="font-mono text-white min-w-[3ch] text-center">{holeDistanceM.toFixed(1)}m</span>
        <button 
          onClick={() => setHoleDistanceM(holeDistanceM + 0.5)}
          className="hover:text-sl-green font-bold px-2"
        >+</button>
      </div>
    </div>
  );
};
