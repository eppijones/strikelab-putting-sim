import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { usePuttingState } from '../../contexts/WebSocketContext';

export const NeuralFeedback: React.FC = () => {
  return null; // Neural feedback disabled as per user request
  /*
  const { gameState } = usePuttingState();
  const isProcessing = gameState === 'TRACKING';

  return (
    <AnimatePresence>
      {isProcessing && (
        <motion.div
  ...
        </motion.div>
      )}
    </AnimatePresence>
  );
  */
};
