import React from 'react';
import { usePuttingState } from '../../contexts/WebSocketContext';
import { motion, AnimatePresence } from 'framer-motion';
import { Wind, ArrowUpRight, Maximize } from 'lucide-react';

export const StatsHUD: React.FC = () => {
  const { lastJsonMessage, gameState } = usePuttingState();
  const shot = lastJsonMessage?.shot;
  const isVisible = !!shot && (gameState === 'TRACKING' || gameState === 'VIRTUAL_ROLLING' || gameState === 'STOPPED');

  // usePuttingState provides gameState and shot, we just use them
  
  if (!isVisible || !shot) return null;

  return (
    <AnimatePresence>
      <motion.div 
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 20 }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2 flex gap-4 md:gap-8 bg-sl-panel/90 backdrop-blur-md px-6 py-4 rounded-2xl border border-white/10 shadow-2xl"
      >
        <StatItem 
          icon={<Wind size={18} />} 
          label="Speed" 
          value={shot.speed_m_s.toFixed(2)} 
          unit="m/s" 
          color="text-sl-green"
        />
        <div className="w-px bg-white/10" />
        <StatItem 
          icon={<Maximize size={18} />} 
          label="Distance" 
          value={shot.distance_m.toFixed(2)} 
          unit="m" 
          color="text-sl-cyan"
        />
        <div className="w-px bg-white/10" />
        <StatItem 
          icon={<ArrowUpRight size={18} />} 
          label="Direction" 
          value={Math.abs(shot.direction_deg).toFixed(1)} 
          unit={`° ${shot.direction_deg > 0 ? 'R' : 'L'}`} 
          color="text-sl-accent"
        />
      </motion.div>
    </AnimatePresence>
  );
};

const StatItem: React.FC<{ icon: React.ReactNode, label: string, value: string, unit: string, color: string }> = ({ icon, label, value, unit, color }) => (
  <div className="flex flex-col items-center min-w-[80px]">
    <div className="flex items-center gap-1 text-xs text-gray-400 uppercase tracking-wide mb-1">
      {icon}
      <span>{label}</span>
    </div>
    <div className="flex items-baseline gap-1">
      <span className={`text-3xl font-bold font-mono ${color}`}>{value}</span>
      <span className="text-sm text-gray-500 font-medium">{unit}</span>
    </div>
  </div>
);
