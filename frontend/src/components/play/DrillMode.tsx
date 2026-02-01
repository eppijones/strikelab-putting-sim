import React, { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Target, Trophy, Zap, X, Play, ChevronRight } from 'lucide-react';
import { usePuttingState, type DrillType } from '../../contexts/WebSocketContext';
import { useSound } from '../../hooks/useSoundEffects';

interface DrillModeProps {
  className?: string;
}

const getRatingColor = (rating: string): string => {
  switch (rating) {
    case 'perfect': return 'text-emerald-400';
    case 'great': return 'text-blue-400';
    case 'good': return 'text-yellow-400';
    case 'fair': return 'text-orange-400';
    default: return 'text-red-400';
  }
};

const getRatingEmoji = (rating: string): string => {
  switch (rating) {
    case 'perfect': return '🎯';
    case 'great': return '👍';
    case 'good': return '👌';
    case 'fair': return '😐';
    default: return '😢';
  }
};

export const DrillMode: React.FC<DrillModeProps> = ({ className = '' }) => {
  const { drillData, startDrill, stopDrill, gameState } = usePuttingState();
  const { playSound } = useSound();
  const lastAttemptRef = useRef<string | null>(null);
  
  const isActive = drillData?.active || false;
  const drillType = drillData?.drill_type || 'none';
  
  // Play sound on drill results
  useEffect(() => {
    if (drillData?.last_attempt && gameState === 'STOPPED') {
      const attemptKey = `${drillData.attempts}-${drillData.last_attempt.rating}`;
      if (attemptKey !== lastAttemptRef.current) {
        lastAttemptRef.current = attemptKey;
        
        if (drillData.last_attempt.rating === 'perfect') {
          playSound('hole');
        } else if (drillData.last_attempt.rating === 'miss') {
          playSound('miss');
        } else {
          playSound('click');
        }
      }
    }
  }, [drillData?.last_attempt, drillData?.attempts, gameState, playSound]);

  // If no drill is active, show the drill selection panel
  if (!isActive) {
    return (
      <div className={`bg-slate-800/80 backdrop-blur-md rounded-2xl border border-white/10 p-6 ${className}`}>
        <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          <Zap size={20} className="text-yellow-400" />
          Practice Drills
        </h3>
        
        <div className="space-y-3">
          {/* Distance Control */}
          <button
            onClick={() => startDrill('distance_control')}
            className="w-full flex items-center justify-between p-4 rounded-xl 
                     bg-emerald-500/20 border border-emerald-500/30 
                     hover:bg-emerald-500/30 transition-colors group"
          >
            <div className="flex items-center gap-3">
              <Target size={24} className="text-emerald-400" />
              <div className="text-left">
                <div className="font-semibold text-white">Distance Control</div>
                <div className="text-sm text-white/60">Hit random target distances</div>
              </div>
            </div>
            <ChevronRight size={20} className="text-white/40 group-hover:text-white/80 transition-colors" />
          </button>
          
          {/* Ladder Drill */}
          <button
            onClick={() => startDrill('ladder_drill')}
            className="w-full flex items-center justify-between p-4 rounded-xl 
                     bg-blue-500/20 border border-blue-500/30 
                     hover:bg-blue-500/30 transition-colors group"
          >
            <div className="flex items-center gap-3">
              <Trophy size={24} className="text-blue-400" />
              <div className="text-left">
                <div className="font-semibold text-white">Ladder Drill</div>
                <div className="text-sm text-white/60">1m → 2m → 3m → 4m → 5m → 6m</div>
              </div>
            </div>
            <ChevronRight size={20} className="text-white/40 group-hover:text-white/80 transition-colors" />
          </button>
        </div>
      </div>
    );
  }

  // Active drill display
  return (
    <div className={`bg-slate-800/80 backdrop-blur-md rounded-2xl border border-white/10 overflow-hidden ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-white/10 bg-slate-700/50">
        <div className="flex items-center gap-2">
          {drillType === 'distance_control' ? (
            <Target size={20} className="text-emerald-400" />
          ) : (
            <Trophy size={20} className="text-blue-400" />
          )}
          <span className="font-semibold text-white">
            {drillType === 'distance_control' ? 'Distance Control' : 'Ladder Drill'}
          </span>
        </div>
        <button
          onClick={stopDrill}
          className="p-2 rounded-lg hover:bg-white/10 transition-colors"
          title="Stop drill"
        >
          <X size={18} className="text-white/60" />
        </button>
      </div>
      
      {/* Current target */}
      <div className="p-6 text-center border-b border-white/10">
        <div className="text-sm text-white/50 uppercase tracking-wider mb-2">
          {drillType === 'ladder_drill' ? `Rung ${(drillData?.ladder_position || 0) + 1} of 6` : 'Target Distance'}
        </div>
        <div className="text-5xl font-bold text-white">
          {drillData?.current_target_m?.toFixed(1)}
          <span className="text-2xl text-white/60 ml-1">m</span>
        </div>
        
        {/* Ladder progress bar */}
        {drillType === 'ladder_drill' && (
          <div className="mt-4 flex gap-1">
            {[1, 2, 3, 4, 5, 6].map((rung, i) => (
              <div
                key={rung}
                className={`flex-1 h-2 rounded-full transition-colors ${
                  i < (drillData?.ladder_position || 0)
                    ? 'bg-blue-400'
                    : i === (drillData?.ladder_position || 0)
                    ? 'bg-blue-400/50 animate-pulse'
                    : 'bg-white/10'
                }`}
              />
            ))}
          </div>
        )}
      </div>
      
      {/* Last attempt result */}
      <AnimatePresence mode="wait">
        {drillData?.last_attempt && (
          <motion.div
            key={`${drillData.attempts}`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="p-4 border-b border-white/10"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="text-2xl">
                  {getRatingEmoji(drillData.last_attempt.rating)}
                </span>
                <div>
                  <div className={`font-bold uppercase ${getRatingColor(drillData.last_attempt.rating)}`}>
                    {drillData.last_attempt.rating}
                  </div>
                  <div className="text-sm text-white/50">
                    {drillData.last_attempt.error_cm.toFixed(0)}cm off
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-yellow-400">
                  +{drillData.last_attempt.points}
                </div>
                <div className="text-xs text-white/40">points</div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Stats */}
      <div className="p-4 grid grid-cols-3 gap-4">
        <div className="text-center">
          <div className="text-2xl font-bold text-yellow-400">
            {drillData?.total_points || 0}
          </div>
          <div className="text-xs text-white/40 uppercase">Points</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-white">
            {drillData?.attempts || 0}
          </div>
          <div className="text-xs text-white/40 uppercase">Attempts</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-white">
            {Math.floor((drillData?.duration_s || 0) / 60)}:{String(Math.floor((drillData?.duration_s || 0) % 60)).padStart(2, '0')}
          </div>
          <div className="text-xs text-white/40 uppercase">Time</div>
        </div>
      </div>
    </div>
  );
};
