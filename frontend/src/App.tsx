import React, { useState, useEffect, useRef } from 'react';
import { WebSocketProvider, usePuttingState } from './contexts/WebSocketContext';
import { ModeToggle, type AppMode } from './components/shared/ModeToggle';
import { ConnectionStatus } from './components/shared/ConnectionStatus';
import { SettingsMenu } from './components/shared/SettingsMenu';
import { PlayMode } from './components/play/PlayMode';
import { LabMode } from './components/lab/LabMode';
import { AnalyticsPanel } from './components/play/AnalyticsPanel';
import { AnimatePresence, motion } from 'framer-motion';
import { SoundProvider, useSound } from './hooks/useSoundEffects';
import { BarChart3 } from 'lucide-react';

// Component that listens to game events and plays sounds
const GameSoundManager: React.FC = () => {
  const { gameState, gameData, sessionData } = usePuttingState();
  const { playSound, stopRolling, settings } = useSound();
  const prevGameStateRef = useRef<string>('ARMED');
  const prevResultRef = useRef<string | null>(null);
  const prevStreakRef = useRef<number>(0);
  
  useEffect(() => {
    if (!settings.enabled) return;
    
    const prevState = prevGameStateRef.current;
    
    // Play rolling sound during tracking
    if (gameState === 'TRACKING' && prevState !== 'TRACKING') {
      playSound('rolling', { speed: 0.8 });
    }
    
    // Adjust rolling sound during virtual rolling
    if (gameState === 'VIRTUAL_ROLLING') {
      playSound('rolling', { speed: 0.4 });
    }
    
    // Stop rolling when stopped
    if (gameState === 'STOPPED' && (prevState === 'TRACKING' || prevState === 'VIRTUAL_ROLLING')) {
      stopRolling();
      
      // Play result sound
      const result = gameData?.last_shot?.result;
      if (result && result !== prevResultRef.current) {
        prevResultRef.current = result;
        
        if (result === 'made') {
          playSound('hole');
          
          // Check for streak celebration
          const currentStreak = sessionData?.current_streak || 0;
          if (currentStreak >= 3 && currentStreak > prevStreakRef.current) {
            setTimeout(() => playSound('streak', { streakCount: currentStreak }), 500);
          }
          prevStreakRef.current = currentStreak;
        } else if (result === 'lip_out') {
          playSound('lipOut');
        } else if (result !== 'pending') {
          playSound('miss');
        }
      }
    }
    
    // Reset when armed
    if (gameState === 'ARMED') {
      stopRolling();
      prevResultRef.current = null;
    }
    
    prevGameStateRef.current = gameState;
  }, [gameState, gameData?.last_shot?.result, sessionData?.current_streak, playSound, stopRolling, settings.enabled]);
  
  return null;
};

const AppContent: React.FC = () => {
  const [mode, setMode] = useState<AppMode>('PLAY');
  const [showAnalytics, setShowAnalytics] = useState(false);

  return (
    <div className="relative w-full h-screen overflow-hidden bg-sl-dark text-white selection:bg-sl-green selection:text-sl-dark">
      <GameSoundManager />
      <ConnectionStatus />
      <ModeToggle mode={mode} setMode={setMode} />
      
      {/* Top right controls */}
      <div className="fixed top-4 right-4 z-30 flex gap-2">
        {/* Analytics button */}
        <button
          onClick={() => setShowAnalytics(true)}
          className="p-3 rounded-full bg-white/10 backdrop-blur-md border border-white/20 
                    hover:bg-white/20 transition-all duration-200"
          title="Analytics"
        >
          <BarChart3 size={20} className="text-white/80" />
        </button>
        
        <SettingsMenu />
      </div>
      
      {/* Analytics Panel */}
      <AnalyticsPanel isOpen={showAnalytics} onClose={() => setShowAnalytics(false)} />
      
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
      <SoundProvider>
        <AppContent />
      </SoundProvider>
    </WebSocketProvider>
  );
};

export default App;
