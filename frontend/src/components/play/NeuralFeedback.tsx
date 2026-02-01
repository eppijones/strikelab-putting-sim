import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { usePuttingState } from '../../contexts/WebSocketContext';

export const NeuralFeedback: React.FC = () => {
  const { gameState } = usePuttingState();
  const isProcessing = gameState === 'TRACKING';

  return (
    <AnimatePresence>
      {isProcessing && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none"
        >
          {/* Neural Ring Animation */}
          <div className="relative w-32 h-32 flex items-center justify-center">
            {/* Core */}
            <motion.div 
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ repeat: Infinity, duration: 2 }}
              className="absolute w-4 h-4 bg-neural-cyan rounded-full shadow-[0_0_20px_#00D4FF]"
            />
            
            {/* Orbiting Ring 1 */}
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ repeat: Infinity, duration: 3, ease: "linear" }}
              className="absolute w-16 h-16 border border-neural-cyan/30 rounded-full border-t-neural-cyan"
            />

            {/* Orbiting Ring 2 (Counter) */}
            <motion.div
              animate={{ rotate: -360 }}
              transition={{ repeat: Infinity, duration: 4, ease: "linear" }}
              className="absolute w-24 h-24 border border-neural-violet/20 rounded-full border-b-neural-violet"
            />
            
            {/* Label */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 40 }}
              className="absolute text-[10px] font-mono text-neural-cyan tracking-widest uppercase"
            >
              ANALYZING
            </motion.div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};
