import React, { useState, useEffect } from 'react';
import { WebSocketProvider } from './contexts/WebSocketContext';
import { ModeToggle, type AppMode } from './components/shared/ModeToggle';
import { ConnectionStatus } from './components/shared/ConnectionStatus';
import { PlayMode } from './components/play/PlayMode';
import { LabMode } from './components/lab/LabMode';
import { AnimatePresence, motion } from 'framer-motion';

const AppContent: React.FC = () => {
  const [mode, setMode] = useState<AppMode>('PLAY');

  /* Remove persistence to ensure we always start in PLAY
  useEffect(() => {
    localStorage.setItem('strikeLabMode', mode);
  }, [mode]);
  */

  return (
    <div className="relative w-full h-screen overflow-hidden bg-sl-dark text-white selection:bg-sl-green selection:text-sl-dark">
      <ConnectionStatus />
      <ModeToggle mode={mode} setMode={setMode} />
      
      <AnimatePresence mode="wait">
        <motion.div
          key={mode}
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 1.02 }}
          transition={{ duration: 0.3, ease: "easeInOut" }}
          className="w-full h-full"
        >
          {mode === 'PLAY' ? <PlayMode /> : <LabMode />}
        </motion.div>
      </AnimatePresence>
    </div>
  );
};

const App: React.FC = () => {
  return (
    <WebSocketProvider>
      <AppContent />
    </WebSocketProvider>
  );
};

export default App;
