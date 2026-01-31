import React, { useState } from 'react';
import { usePuttingState } from '../../contexts/WebSocketContext';

export const CameraOverlay: React.FC = () => {
  const { lastJsonMessage, ballPosition, gameState } = usePuttingState();
  const [imgError, setImgError] = useState(false);
  
  // Default to 1280x800 if not yet received
  const width = lastJsonMessage?.resolution[0] || 1280;
  const height = lastJsonMessage?.resolution[1] || 800;
  
  // Overlay radius scale from config (default 1.15)
  const overlayScale = lastJsonMessage?.overlay_radius_scale || 1.15;
  
  // Velocity vector scaling
  const VELOCITY_SCALE = 0.5; 

  const ballVisible = lastJsonMessage?.ball_visible;
  const ballRadius = lastJsonMessage?.ball?.radius_px || 15;
  const velocity = lastJsonMessage?.velocity;
  
  // Prediction trajectory
  const trajectory = lastJsonMessage?.prediction?.trajectory || [];

  return (
    <div className="relative w-full h-full bg-black flex items-center justify-center overflow-hidden border border-sl-panel rounded-lg">
      <div 
        className="relative"
        style={{ aspectRatio: `${width}/${height}`, maxHeight: '100%', maxWidth: '100%' }}
      >
        {/* MJPEG Stream */}
        {!imgError ? (
          <img 
            src="/api/video" 
            alt="Camera Feed"
            className="w-full h-full object-contain"
            onError={() => setImgError(true)}
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center bg-gray-900 text-gray-500">
            <span className="text-sm">Camera feed unavailable</span>
          </div>
        )}

        {/* Tracking Overlay */}
        <svg 
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
          viewBox={`0 0 ${width} ${height}`}
          preserveAspectRatio="xMidYMid meet"
        >
          {/* Prediction Trajectory */}
          {trajectory.length > 0 && (
            <polyline
              points={trajectory.map(p => `${p[0]},${p[1]}`).join(' ')}
              fill="none"
              stroke="#f59e0b" // sl-accent
              strokeWidth="2"
              strokeDasharray="5,5"
              opacity="0.6"
            />
          )}

          {/* Ball Detection Circle */}
          {ballPosition && ballVisible && (
            <g transform={`translate(${ballPosition.x}, ${ballPosition.y})`}>
              {/* Outer detection ring */}
              <circle 
                r={ballRadius * overlayScale} 
                fill="none" 
                stroke={gameState === 'TRACKING' ? '#ef4444' : '#22c55e'} 
                strokeWidth="2"
              />
              
              {/* Center Crosshair */}
              <line x1="-10" y1="0" x2="10" y2="0" stroke="white" strokeWidth="1" opacity="0.5" />
              <line x1="0" y1="-10" x2="0" y2="10" stroke="white" strokeWidth="1" opacity="0.5" />

              {/* Velocity Vector */}
              {velocity && velocity.speed_px_s > 10 && (
                <line 
                  x1="0" 
                  y1="0" 
                  x2={velocity.vx_px_s * VELOCITY_SCALE} 
                  y2={velocity.vy_px_s * VELOCITY_SCALE} 
                  stroke="#06b6d4" // sl-cyan
                  strokeWidth="2"
                  markerEnd="url(#arrowhead)"
                />
              )}
            </g>
          )}

          {/* Definitions */}
          <defs>
            <marker
              id="arrowhead"
              markerWidth="10"
              markerHeight="7"
              refX="9"
              refY="3.5"
              orient="auto"
            >
              <polygon points="0 0, 10 3.5, 0 7" fill="#06b6d4" />
            </marker>
          </defs>
        </svg>

        {/* State Badge */}
        <div className="absolute top-2 left-2 px-2 py-1 bg-black/60 rounded text-xs font-mono border border-white/10">
          <span className={gameState === 'TRACKING' ? 'text-red-500' : 'text-green-500'}>
            {gameState}
          </span>
          <span className="ml-2 text-gray-400">
            {width}x{height}
          </span>
        </div>
      </div>
    </div>
  );
};
