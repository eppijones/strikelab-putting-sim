import React from 'react';
import { usePuttingState } from '../../contexts/WebSocketContext';

interface DistanceControlProps {
  className?: string;
}

export const DistanceControl: React.FC<DistanceControlProps> = ({ className = '' }) => {
  const { gameData, setHoleDistance } = usePuttingState();
  const currentDistance = gameData?.hole?.distance_m || 3;

  // Short putts (tap-ins to close range) and longer putts
  const presets = [0.5, 1, 2, 3, 5, 8, 10, 15, 20, 25];

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setHoleDistance(parseFloat(e.target.value));
  };

  return (
    <div className={`bg-black/60 backdrop-blur-md rounded-xl border border-white/10 p-4 text-white ${className}`}>
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-sm font-bold font-mono text-white/80">HOLE DISTANCE</h3>
        <span className="text-emerald-400 font-mono font-bold">{currentDistance.toFixed(1)}m</span>
      </div>

      <div className="mb-4">
        <input
          type="range"
          min="0.5"
          max="25"
          step="0.1"
          value={currentDistance}
          onChange={handleSliderChange}
          className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer accent-emerald-500"
        />
        <div className="flex justify-between text-xs text-white/30 mt-1 font-mono">
          <span>0.5m</span>
          <span>25m</span>
        </div>
      </div>

      <div className="grid grid-cols-5 gap-1.5">
        {presets.map((dist) => (
          <button
            key={dist}
            onClick={() => setHoleDistance(dist)}
            className={`px-1 py-1.5 rounded-lg text-xs font-mono font-bold transition-all ${
              Math.abs(currentDistance - dist) < 0.05
                ? 'bg-emerald-500 text-white shadow-[0_0_10px_rgba(16,185,129,0.4)]'
                : 'bg-white/5 hover:bg-white/10 text-white/60 hover:text-white'
            }`}
          >
            {dist < 1 ? `${dist * 100}cm` : `${dist}m`}
          </button>
        ))}
      </div>
    </div>
  );
};
