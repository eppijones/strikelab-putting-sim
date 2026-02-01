import React from 'react';
import { usePuttingState } from '../../contexts/WebSocketContext';
import { motion, AnimatePresence } from 'framer-motion';
import { Wind, ArrowUpRight, Maximize } from 'lucide-react';

export const StatsHUD: React.FC = () => {
  const { lastJsonMessage, gameState } = usePuttingState();
  const shot = lastJsonMessage?.shot;
  const isVisible = !!shot && (gameState === 'TRACKING' || gameState === 'VIRTUAL_ROLLING' || gameState === 'STOPPED');

  if (!isVisible || !shot) return null;

  return (
    <AnimatePresence>
      <motion.div 
        initial={{ opacity: 0, y: 100 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 50 }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2 flex gap-4 bg-white/60 backdrop-blur-[16px] px-8 py-5 rounded-[24px] border border-white/80 shadow-glass"
      >
        <StatItem 
          icon={<Wind size={16} />} 
          label="Speed" 
          value={shot.speed_m_s.toFixed(2)} 
          unit="m/s" 
          color="text-nordic-forest"
        />
        <div className="w-px bg-nordic-forest/10" />
        <StatItem 
          icon={<Maximize size={16} />} 
          label="Distance" 
          value={shot.distance_m.toFixed(2)} 
          unit="m" 
          color="text-nordic-forest"
        />
        <div className="w-px bg-nordic-forest/10" />
        <StatItem 
          icon={<ArrowUpRight size={16} />} 
          label="Line" 
          value={Math.abs(shot.direction_deg).toFixed(1)} 
          unit={`° ${shot.direction_deg > 0 ? 'R' : 'L'}`} 
          color={Math.abs(shot.direction_deg) < 1.0 ? "text-nordic-sage" : "text-nordic-forest"}
        />
      </motion.div>
    </AnimatePresence>
  );
};

const StatItem: React.FC<{ icon: React.ReactNode, label: string, value: string, unit: string, color: string }> = ({ icon, label, value, unit, color }) => (
  <div className="flex flex-col items-center min-w-[100px]">
    <div className="flex items-center gap-1 text-[10px] text-nordic-forest/60 uppercase tracking-widest font-sans font-bold mb-1">
      {icon}
      <span>{label}</span>
    </div>
    <div className="flex items-baseline gap-1">
      <span className={`text-4xl font-mono font-bold tracking-tighter ${color}`}>{value}</span>
      <span className="text-xs text-nordic-forest/40 font-medium">{unit}</span>
    </div>
  </div>
);

