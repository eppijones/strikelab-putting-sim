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
    <div className="fixed inset-0 w-full h-full overflow-hidden bg-[#F2F0EB] text-white selection:bg-sl-green selection:text-sl-dark">
      <GameSoundManager />
      <ConnectionStatus />
      {/* Top right controls */}
      <div className="fixed top-4 right-4 z-50 flex items-center gap-3">
        {/* Analytics button */}
        <button
          onClick={() => setShowAnalytics(true)}
          className="p-3 rounded-full bg-white/80 backdrop-blur-md border border-slate-200 
                    hover:bg-white transition-all duration-200 shadow-sm"
          title="Analytics"
        >
          <BarChart3 size={20} className="text-slate-600" />
        </button>
        
        <SettingsMenu />
        
        <div className="w-px h-8 bg-slate-200 mx-1" />
        
        <ModeToggle mode={mode} setMode={setMode} />
      </div>
      
      {/* Analytics Panel */}
      <AnalyticsPanel isOpen={showAnalytics} onClose={() => setShowAnalytics(false)} />
      
      <AnimatePresence mode="wait">
        <motion.div
          key={mode}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
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
