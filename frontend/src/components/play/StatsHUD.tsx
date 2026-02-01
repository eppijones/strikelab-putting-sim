import React, { useState, useEffect } from 'react';
import { usePuttingState, type ShotResult } from '../../contexts/WebSocketContext';
import { motion, AnimatePresence } from 'framer-motion';
import { Wind, ArrowUpRight, Maximize, Target } from 'lucide-react';

export const StatsHUD: React.FC = () => {
  const { lastJsonMessage, gameState } = usePuttingState();
  const [displayShot, setDisplayShot] = useState<ShotResult | null>(null);

  // Update display shot whenever we get new valid shot data
  useEffect(() => {
    if (lastJsonMessage?.shot) {
      setDisplayShot(lastJsonMessage.shot);
    }
  }, [lastJsonMessage]);

  // If we haven't seen any shot yet, don't show the HUD
  if (!displayShot) return null;

  const isMeasuring = gameState === 'TRACKING' || gameState === 'VIRTUAL_ROLLING';
  const isReady = gameState === 'ARMED';

  // Use current live data if tracking/rolling, otherwise use stored last shot
  // Note: if live data is null during tracking (shouldn't happen if logic is correct), fallback to last shot
  const activeShot = (isMeasuring && lastJsonMessage?.shot) ? lastJsonMessage.shot : displayShot;

  return (
    <AnimatePresence>
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="absolute top-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
      >
        {/* Main HUD */}
        <div className="flex gap-4 bg-white/60 backdrop-blur-[16px] px-8 py-5 rounded-[24px] border border-white/80 shadow-glass relative">
          
          <StatItem 
            icon={<Wind size={16} />} 
            label="Speed" 
            value={activeShot.speed_m_s.toFixed(2)} 
            unit="m/s" 
            color="text-nordic-forest"
          />
          
          <div className="w-px bg-nordic-forest/10" />
          
          <StatItem 
            icon={<Maximize size={16} />} 
            label="Distance" 
            value={isMeasuring ? <span className="text-base text-nordic-forest/40 font-bold animate-pulse">MEASURING</span> : activeShot.distance_m.toFixed(2)}
            unit={!isMeasuring ? "m" : ""}
            color="text-nordic-forest"
            minWidth="min-w-[140px]"
          />
          
          <div className="w-px bg-nordic-forest/10" />
          
          <StatItem 
            icon={<ArrowUpRight size={16} />} 
            label="Line" 
            value={Math.abs(activeShot.direction_deg).toFixed(1)} 
            unit={`° ${activeShot.direction_deg > 0 ? 'R' : 'L'}`} 
            color={Math.abs(activeShot.direction_deg) < 1.0 ? "text-nordic-sage" : "text-nordic-forest"}
          />
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

const StatItem: React.FC<{ icon: React.ReactNode, label: string, value: string | React.ReactNode, unit: string, color: string, minWidth?: string }> = ({ icon, label, value, unit, color, minWidth = "min-w-[100px]" }) => (
  <div className={`flex flex-col items-center ${minWidth}`}>
    <div className="flex items-center gap-1 text-[10px] text-nordic-forest/60 uppercase tracking-widest font-sans font-bold mb-1">
      {icon}
      <span>{label}</span>
    </div>
    <div className="flex items-baseline gap-1 h-[40px]">
      {typeof value === 'string' ? (
        <span className={`text-4xl font-mono font-bold tracking-tighter ${color}`}>{value}</span>
      ) : (
        <div className="flex items-center h-full">{value}</div>
      )}
      <span className="text-xs text-nordic-forest/40 font-medium">{unit}</span>
    </div>
  </div>
);
