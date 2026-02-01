import React, { useEffect, useState, useRef } from 'react';
import { usePuttingState, type ShotResultType } from '../../contexts/WebSocketContext';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle2, AlertCircle, ArrowDown, ArrowUp, ArrowLeft, ArrowRight, Flag } from 'lucide-react';

interface FeedbackDisplay {
  text: string;
  subtext?: string;
  color: string;
  icon: React.ReactNode;
}

const getFeedbackDisplay = (result: ShotResultType, missDescription: string): FeedbackDisplay => {
  switch (result) {
    case 'made':
      return {
        text: 'PERFECT',
        subtext: 'Great putt!',
        icon: <Flag className="w-8 h-8" />,
        color: 'text-emerald-600',
      };
    case 'lip_out':
      return {
        text: 'LIP OUT',
        subtext: 'So close!',
        icon: <AlertCircle className="w-8 h-8" />,
        color: 'text-amber-500',
      };
    case 'miss_short':
      return {
        text: 'SHORT',
        subtext: missDescription,
        icon: <ArrowDown className="w-8 h-8" />,
        color: 'text-blue-500',
      };
    case 'miss_long':
      return {
        text: 'LONG',
        subtext: missDescription,
        icon: <ArrowUp className="w-8 h-8" />,
        color: 'text-orange-500',
      };
    case 'miss_left':
      return {
        text: 'MISS LEFT',
        subtext: missDescription,
        icon: <ArrowLeft className="w-8 h-8" />,
        color: 'text-rose-500',
      };
    case 'miss_right':
      return {
        text: 'MISS RIGHT',
        subtext: missDescription,
        icon: <ArrowRight className="w-8 h-8" />,
        color: 'text-rose-500',
      };
    default:
      return {
        text: '',
        icon: null,
        color: 'text-gray-400',
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
        
        // Hide after 3 seconds
        hideTimeoutRef.current = setTimeout(() => {
          setShowFeedback(false);
        }, 3000);
      }
    }
    
    // Clear feedback when returning to ARMED or READY
    if (gameState === 'ARMED') {
      setShowFeedback(false);
      if (hideTimeoutRef.current) {
        clearTimeout(hideTimeoutRef.current);
      }
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
          initial={{ scale: 0.9, opacity: 0, x: 20 }}
          animate={{ scale: 1, opacity: 1, x: 0 }}
          exit={{ scale: 0.95, opacity: 0, x: 20 }}
          transition={{ type: 'spring', stiffness: 300, damping: 25 }}
          className="absolute bottom-72 right-8 flex flex-col items-end z-50 pointer-events-none"
        >
          {/* Glass Card */}
          <div className="bg-white/80 backdrop-blur-xl px-8 py-6 rounded-[24px] border border-white/60 shadow-glass flex flex-col items-center gap-3 min-w-[200px]">
            
            {/* Icon Circle */}
            <div className={`p-3 rounded-full bg-white/50 shadow-sm ${feedback.color}`}>
              {feedback.icon}
            </div>
            
            <div className="flex flex-col items-center gap-0.5">
              {/* Main text */}
              <div className={`text-2xl font-black ${feedback.color} text-center tracking-tight font-sans`}>
                {feedback.text}
              </div>
              
              {/* Subtext */}
              {feedback.subtext && (
                <div className="text-sm text-nordic-forest/60 text-center font-medium font-mono">
                  {feedback.subtext}
                </div>
              )}
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};
