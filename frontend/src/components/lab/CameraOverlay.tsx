import React, { useRef, useEffect, useState, useCallback } from 'react';
import { usePuttingState } from '../../contexts/WebSocketContext';
import { RefreshCw } from 'lucide-react';

const ReconnectingMjpeg: React.FC<{
  src: string;
  alt: string;
  className?: string;
}> = ({ src, alt, className }) => {
  const imgRef = useRef<HTMLImageElement>(null);
  const [reconnecting, setReconnecting] = useState(false);
  const [activeSrc, setActiveSrc] = useState(src);
  const lastLoadedAt = useRef(Date.now());
  const retryCount = useRef(0);

  const reconnect = useCallback(() => {
    retryCount.current += 1;
    setReconnecting(true);
    const sep = src.includes('?') ? '&' : '?';
    setActiveSrc(`${src}${sep}_t=${Date.now()}`);
  }, [src]);

  useEffect(() => {
    setActiveSrc(src);
    lastLoadedAt.current = Date.now();
    retryCount.current = 0;
  }, [src]);

  useEffect(() => {
    const interval = setInterval(() => {
      const img = imgRef.current;
      if (!img) return;
      if (img.naturalWidth > 0) {
        lastLoadedAt.current = Date.now();
        if (reconnecting) setReconnecting(false);
        retryCount.current = 0;
        return;
      }
      const staleSec = (Date.now() - lastLoadedAt.current) / 1000;
      if (staleSec > 4) reconnect();
    }, 2000);
    return () => clearInterval(interval);
  }, [reconnect, reconnecting]);

  return (
    <div className="relative w-full h-full">
      <img
        ref={imgRef}
        src={activeSrc}
        alt={alt}
        className={className}
        onError={() => setTimeout(reconnect, 1000 + Math.min(retryCount.current * 500, 3000))}
        onLoad={() => {
          lastLoadedAt.current = Date.now();
          if (reconnecting) setReconnecting(false);
          retryCount.current = 0;
        }}
      />
      {reconnecting && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/60">
          <div className="flex items-center gap-2 text-white/60 text-xs">
            <RefreshCw size={14} className="animate-spin" />
            <span>Reconnecting...</span>
          </div>
        </div>
      )}
    </div>
  );
};

export const CameraOverlay: React.FC = () => {
  const { lastJsonMessage, ballPosition, gameState } = usePuttingState();
  
  const width = lastJsonMessage?.resolution?.[0] ?? 1280;
  const height = lastJsonMessage?.resolution?.[1] ?? 800;
  
  const overlayScale = lastJsonMessage?.overlay_radius_scale || 1.15;
  
  const VELOCITY_SCALE = 0.5; 

  const ballVisible = lastJsonMessage?.ball_visible;
  const ballRadius = lastJsonMessage?.ball?.radius_px || 15;
  const velocity = lastJsonMessage?.velocity;
  
  const trajectory = lastJsonMessage?.prediction?.trajectory || [];

  return (
    <div className="relative w-full h-full bg-black flex items-center justify-center overflow-hidden border border-sl-panel rounded-lg">
      <div 
        className="relative"
        style={{ aspectRatio: `${width}/${height}`, maxHeight: '100%', maxWidth: '100%' }}
      >
        <ReconnectingMjpeg
          src="/api/video"
          alt="Camera Feed"
          className="w-full h-full object-contain"
        />

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
