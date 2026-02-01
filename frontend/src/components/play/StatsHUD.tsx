import React, { useState, useEffect } from 'react';
import { usePuttingState, type ShotResult } from '../../contexts/WebSocketContext';
import { motion, AnimatePresence } from 'framer-motion';
import { Wind, ArrowUpRight, Maximize, Target, Trophy, Flame } from 'lucide-react';

export const StatsHUD: React.FC = () => {
  const { lastJsonMessage, gameState, sessionData, gameData } = usePuttingState();
  const [displayShot, setDisplayShot] = useState<ShotResult | null>(null);

  // Update display shot whenever we get new valid shot data
  useEffect(() => {
    if (lastJsonMessage?.shot) {
      setDisplayShot(lastJsonMessage.shot);
    }
  }, [lastJsonMessage]);

  const isMeasuring = gameState === 'TRACKING' || gameState === 'VIRTUAL_ROLLING';

  // Use current live data if tracking/rolling, otherwise use stored last shot
  const activeShot = (isMeasuring && lastJsonMessage?.shot) ? lastJsonMessage.shot : displayShot;

  // Get session stats
  const totalPutts = sessionData?.total_putts || 0;
  const puttsMade = sessionData?.putts_made || 0;
  const makePercentage = sessionData?.make_percentage || 0;
  const currentStreak = sessionData?.current_streak || 0;
  const bestStreak = sessionData?.best_streak || 0;
  const holeDistance = gameData?.hole?.distance_m || 3.0;

  return (
    <AnimatePresence>
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="absolute top-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-3"
      >
        {/* Session Stats Bar (always visible) */}
        <div className="flex gap-3 items-center">
          {/* Score */}
          <div className="flex items-center gap-2 bg-emerald-500/20 backdrop-blur-md px-4 py-2 rounded-full border border-emerald-400/30">
            <Trophy size={14} className="text-emerald-400" />
            <span className="text-sm font-bold text-emerald-400">
              {puttsMade}/{totalPutts}
            </span>
            {totalPutts > 0 && (
              <span className="text-xs text-emerald-400/70">
                ({makePercentage.toFixed(0)}%)
              </span>
            )}
          </div>
          
          {/* Streak */}
          {currentStreak > 0 && (
            <motion.div 
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="flex items-center gap-2 bg-orange-500/20 backdrop-blur-md px-4 py-2 rounded-full border border-orange-400/30"
            >
              <Flame size={14} className="text-orange-400" />
              <span className="text-sm font-bold text-orange-400">
                {currentStreak} in a row
              </span>
            </motion.div>
          )}
          
          {/* Hole Distance */}
          <div className="flex items-center gap-2 bg-white/40 backdrop-blur-md px-4 py-2 rounded-full border border-white/30">
            <Target size={14} className="text-nordic-forest/70" />
            <span className="text-sm font-medium text-nordic-forest/70">
              {holeDistance.toFixed(1)}m
            </span>
          </div>
        </div>

        {/* Main Shot Stats HUD (only visible after first shot) */}
        {activeShot && (
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
        )}
        
        {/* Best streak badge */}
        {bestStreak >= 3 && !currentStreak && (
          <div className="text-xs text-nordic-forest/50">
            Best streak: {bestStreak}
          </div>
        )}
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
