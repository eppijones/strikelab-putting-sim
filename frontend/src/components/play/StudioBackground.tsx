import React from 'react';

export const StudioBackground: React.FC = () => {
  return (
    <div className="absolute inset-0 w-full h-full -z-50 bg-[#F2F0EB] overflow-hidden">
      {/* 1. Base Gradient (Subtle Warmth) */}
      <div className="absolute top-0 left-0 w-full h-2/3 bg-gradient-to-b from-[#D4C5A8]/20 to-transparent opacity-40 pointer-events-none" />

      {/* 2. Grid Texture (Darker for contrast on light bg) */}
      <div 
        className="absolute inset-0 opacity-10 pointer-events-none"
        style={{ 
          backgroundImage: `
            linear-gradient(rgba(30, 58, 43, 0.2) 1px, transparent 1px),
            linear-gradient(90deg, rgba(30, 58, 43, 0.2) 1px, transparent 1px)
          `,
          backgroundSize: '40px 40px',
          maskImage: 'radial-gradient(circle at center, black 40%, transparent 100%)'
        }}
      />
    </div>
  );
};
