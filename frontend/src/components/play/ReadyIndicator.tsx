import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { usePuttingState } from '../../contexts/WebSocketContext';

export const ReadyIndicator: React.FC = () => {
  const { gameState } = usePuttingState();
  const isReady = gameState === 'ARMED';

  return (
    <AnimatePresence>
      {isReady && (
        <motion.div
          initial={{ opacity: 0, y: 20, scale: 0.9 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 10, scale: 0.95 }}
          transition={{ type: "spring", stiffness: 300, damping: 20 }}
          className="absolute bottom-[10%] left-[49.7%] -translate-x-1/2 z-10 flex flex-col items-center"
        >
          <div className="bg-white/60 backdrop-blur-[16px] text-nordic-forest text-sm font-bold px-8 py-3 rounded-[24px] uppercase tracking-[0.2em] shadow-glass animate-pulse border border-white/80">
            Ready for Shot
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};
