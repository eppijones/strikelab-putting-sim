import React, { useEffect, useState } from 'react';
import { usePuttingState } from '../../contexts/WebSocketContext';
import { motion, AnimatePresence } from 'framer-motion';
import confetti from 'canvas-confetti';

export const ShotFeedback: React.FC = () => {
  const { lastJsonMessage, gameState } = usePuttingState();
  const [showResult, setShowResult] = useState<string | null>(null);

  useEffect(() => {
    // Simple logic for demonstration: 
    // In a real app, we'd check distance to hole radius.
    // For now, let's just use the 'Stopped' state to trigger a generic "Finish"
    // Ideally, backend calculates 'Result' (Hole In, Miss Short, Miss Left, etc.)
    
    if (gameState === 'STOPPED' && lastJsonMessage?.shot) {
      // Simple feedback
      setShowResult("STOPPED");
      
      // Fire confetti for fun
      confetti({
        particleCount: 100,
        spread: 70,
        origin: { y: 0.6 },
        colors: ['#22c55e', '#06b6d4', '#ffffff']
      });

      // Clear after 3s
      const timer = setTimeout(() => setShowResult(null), 3000);
      return () => clearTimeout(timer);
    } else if (gameState === 'ARMED') {
      setShowResult(null);
    }
  }, [gameState, lastJsonMessage]);

  return (
    <AnimatePresence>
      {showResult && (
        <motion.div
          initial={{ scale: 0.5, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 1.5, opacity: 0 }}
          className="absolute top-1/3 left-1/2 -translate-x-1/2 text-6xl font-black text-sl-neon italic drop-shadow-[0_0_15px_rgba(74,222,128,0.5)]"
        >
          {showResult}
        </motion.div>
      )}
    </AnimatePresence>
  );
};
