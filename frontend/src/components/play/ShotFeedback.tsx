import React, { useEffect, useState, useRef } from 'react';
import { usePuttingState, type ShotResultType } from '../../contexts/WebSocketContext';
import { motion, AnimatePresence } from 'framer-motion';

interface FeedbackDisplay {
  text: string;
  subtext?: string;
  color: string;
  bgColor: string;
  icon: string;
}

const getFeedbackDisplay = (result: ShotResultType, missDescription: string): FeedbackDisplay => {
  switch (result) {
    case 'made':
      return {
        text: 'MADE IT!',
        icon: '⛳',
        color: 'text-emerald-400',
        bgColor: 'bg-emerald-500/20'
      };
    case 'lip_out':
      return {
        text: 'LIP OUT',
        subtext: 'So close!',
        icon: '😮',
        color: 'text-amber-400',
        bgColor: 'bg-amber-500/20'
      };
    case 'miss_short':
      return {
        text: 'SHORT',
        subtext: missDescription,
        icon: '📏',
        color: 'text-blue-400',
        bgColor: 'bg-blue-500/20'
      };
    case 'miss_long':
      return {
        text: 'LONG',
        subtext: missDescription,
        icon: '🎯',
        color: 'text-orange-400',
        bgColor: 'bg-orange-500/20'
      };
    case 'miss_left':
      return {
        text: 'MISS LEFT',
        subtext: missDescription,
        icon: '⬅️',
        color: 'text-purple-400',
        bgColor: 'bg-purple-500/20'
      };
    case 'miss_right':
      return {
        text: 'MISS RIGHT',
        subtext: missDescription,
        icon: '➡️',
        color: 'text-rose-400',
        bgColor: 'bg-rose-500/20'
      };
    default:
      return {
        text: '',
        icon: '',
        color: 'text-gray-400',
        bgColor: 'bg-gray-500/20'
      };
  }
};

export const ShotFeedback: React.FC = () => {
  const { gameState, gameData } = usePuttingState();
  const [feedback, setFeedback] = useState<FeedbackDisplay | null>(null);
  const [showFeedback, setShowFeedback] = useState(false);
  const lastResultRef = useRef<string | null>(null);
  const hideTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    // Show feedback when shot stops and we have a result
    if (gameState === 'STOPPED' && gameData?.last_shot) {
      const result = gameData.last_shot.result;
      const resultKey = `${result}-${gameData.last_shot.distance_to_hole_m}`;
      
      // Only show if this is a new result
      if (result && result !== 'pending' && resultKey !== lastResultRef.current) {
        lastResultRef.current = resultKey;
        const display = getFeedbackDisplay(result, gameData.last_shot.miss_description);
        setFeedback(display);
        setShowFeedback(true);
        
        // Clear any existing timeout
        if (hideTimeoutRef.current) {
          clearTimeout(hideTimeoutRef.current);
        }
        
        // Hide after 2.5 seconds
        hideTimeoutRef.current = setTimeout(() => {
          setShowFeedback(false);
        }, 2500);
      }
    }
    
    // Clear feedback when returning to ARMED
    if (gameState === 'ARMED') {
      // Don't immediately clear - let it fade naturally
    }
    
    return () => {
      if (hideTimeoutRef.current) {
        clearTimeout(hideTimeoutRef.current);
      }
    };
  }, [gameState, gameData?.last_shot]);

  return (
    <AnimatePresence>
      {showFeedback && feedback && feedback.text && (
        <motion.div
          initial={{ scale: 0.5, opacity: 0, y: 20 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.8, opacity: 0, y: -20 }}
          transition={{ type: 'spring', stiffness: 300, damping: 25 }}
          className="absolute top-1/3 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
        >
          {/* Main feedback card */}
          <div className={`${feedback.bgColor} backdrop-blur-xl px-8 py-6 rounded-3xl border border-white/20 shadow-2xl`}>
            {/* Icon */}
            <div className="text-5xl text-center mb-2">
              {feedback.icon}
            </div>
            
            {/* Main text */}
            <div className={`text-5xl font-black ${feedback.color} text-center tracking-tight`}>
              {feedback.text}
            </div>
            
            {/* Subtext */}
            {feedback.subtext && (
              <div className="text-lg text-white/70 text-center mt-2 font-medium">
                {feedback.subtext}
              </div>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};
