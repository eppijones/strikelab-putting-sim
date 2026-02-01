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
          className="absolute bottom-[20%] left-1/2 -translate-x-1/2 z-10"
        >
          <div className="bg-nordic-sage text-white text-xs font-bold px-6 py-2 rounded-full uppercase tracking-[0.2em] shadow-lg animate-pulse border border-white/20 backdrop-blur-sm">
            Ready for Shot
          </div>
          {/* Optional decorative line pointing down to ball */}
          <div className="w-[1px] h-8 bg-gradient-to-b from-nordic-sage/50 to-transparent mx-auto mt-1" />
        </motion.div>
      )}
    </AnimatePresence>
  );
};
