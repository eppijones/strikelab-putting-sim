import React from 'react';

export const StudioBackground: React.FC = () => {
  return (
    <div className="absolute inset-0 w-full h-full -z-50 bg-midnight-surface overflow-hidden">
      {/* 1. Base Gradient (Aurora) */}
      <div className="absolute top-0 left-0 w-full h-2/3 bg-gradient-aurora opacity-60 pointer-events-none" />

      {/* 2. Noise Texture */}
      <div className="absolute inset-0 opacity-[0.03] pointer-events-none mix-blend-overlay"
           style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")` }}
      />

      {/* 3. Dot Grid (Spatial Reference) */}
      <div 
        className="absolute inset-0 opacity-20 pointer-events-none"
        style={{ 
          backgroundImage: 'radial-gradient(rgba(255,255,255,0.3) 1px, transparent 1px)',
          backgroundSize: '40px 40px',
          maskImage: 'linear-gradient(to bottom, transparent, black 20%, black 80%, transparent)'
        }}
      />
    </div>
  );
};
