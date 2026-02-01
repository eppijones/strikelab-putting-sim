import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Settings, X, Volume2, VolumeX, Target, RotateCcw, ChevronDown, Gauge, Eye } from 'lucide-react';
import { usePuttingState } from '../../contexts/WebSocketContext';
import { useSound } from '../../hooks/useSoundEffects';

// Green speed presets
const GREEN_SPEED_PRESETS = [
  { id: 'fast', label: 'Fast', description: 'Tournament (Stimp 12-14)', decel: 0.45 },
  { id: 'medium-fast', label: 'Medium-Fast', description: 'Club championship (10-12)', decel: 0.55 },
  { id: 'medium', label: 'Medium', description: 'Standard club (8-10)', decel: 0.65 },
  { id: 'medium-slow', label: 'Medium-Slow', description: 'Casual play (6-8)', decel: 0.80 },
  { id: 'slow', label: 'Slow', description: 'Practice mat (4-6)', decel: 1.00 },
];

interface SettingsMenuProps {
  className?: string;
}

export const SettingsMenu: React.FC<SettingsMenuProps> = ({ className = '' }) => {
  const [isOpen, setIsOpen] = useState(false);
  const { gameData, setHoleDistance, resetSession, showDistance, setShowDistance } = usePuttingState();
  const { settings: soundSettings, toggleSound, setVolume, playSound } = useSound();
  
  const currentHoleDistance = gameData?.hole?.distance_m || 3.0;
  const [localDistance, setLocalDistance] = useState(currentHoleDistance);
  const [greenSpeed, setGreenSpeed] = useState<string>('medium');
  
  // Fetch current green speed on mount
  useEffect(() => {
    const fetchGreenSpeed = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/green-speed');
        if (res.ok) {
          const data = await res.json();
          if (data.current_preset) {
            setGreenSpeed(data.current_preset);
          }
        }
      } catch (e) {
        console.error('Failed to fetch green speed:', e);
      }
    };
    fetchGreenSpeed();
  }, []);
  
  const handleGreenSpeedChange = async (preset: string) => {
    setGreenSpeed(preset);
    try {
      await fetch('http://localhost:8000/api/green-speed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ preset })
      });
      playSound('click');
    } catch (e) {
      console.error('Failed to set green speed:', e);
    }
  };
  
  const handleDistanceChange = (value: number) => {
    setLocalDistance(value);
  };
  
  const handleDistanceCommit = () => {
    setHoleDistance(localDistance);
    playSound('click');
  };
  
  const handleResetSession = () => {
    if (window.confirm('Reset session? This will clear all statistics for this session.')) {
      resetSession();
      playSound('click');
    }
  };
  
  const presetDistances = [2, 3, 5, 8, 10, 15, 20, 25];
  
  return (
    <>
      {/* Settings button */}
      <button
        onClick={() => {
          setIsOpen(!isOpen);
          playSound('click');
        }}
        className={`p-3 rounded-full bg-white/80 backdrop-blur-md border border-slate-200 
                    hover:bg-white transition-all duration-200 shadow-sm ${className}`}
        title="Settings"
      >
        <Settings size={20} className="text-slate-600" />
      </button>
      
      {/* Settings panel */}
      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40"
              onClick={() => setIsOpen(false)}
            />
            
            {/* Panel */}
            <motion.div
              initial={{ opacity: 0, x: 300 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 300 }}
              transition={{ type: 'spring', damping: 25, stiffness: 300 }}
              className="fixed right-0 top-0 h-full w-80 bg-slate-900/95 backdrop-blur-xl border-l border-white/10 z-50 overflow-y-auto"
            >
              {/* Header */}
              <div className="flex items-center justify-between p-6 border-b border-white/10">
                <h2 className="text-xl font-bold text-white">Settings</h2>
                <button
                  onClick={() => setIsOpen(false)}
                  className="p-2 rounded-full hover:bg-white/10 transition-colors"
                >
                  <X size={20} className="text-white/60" />
                </button>
              </div>
              
              {/* Content */}
              <div className="p-6 space-y-8">
                {/* Hole Distance */}
                <section>
                  <div className="flex items-center gap-2 mb-4">
                    <Target size={18} className="text-emerald-400" />
                    <h3 className="text-sm font-semibold text-white/80 uppercase tracking-wider">
                      Hole Distance
                    </h3>
                  </div>
                  
                  {/* Current value */}
                  <div className="text-center mb-4">
                    <span className="text-4xl font-bold text-white">
                      {localDistance.toFixed(1)}
                    </span>
                    <span className="text-lg text-white/60 ml-1">m</span>
                  </div>
                  
                  {/* Slider */}
                  <input
                    type="range"
                    min="1"
                    max="25"
                    step="0.1"
                    value={localDistance}
                    onChange={(e) => handleDistanceChange(parseFloat(e.target.value))}
                    onMouseUp={handleDistanceCommit}
                    onTouchEnd={handleDistanceCommit}
                    className="w-full h-2 bg-white/10 rounded-full appearance-none cursor-pointer
                               [&::-webkit-slider-thumb]:appearance-none
                               [&::-webkit-slider-thumb]:w-5
                               [&::-webkit-slider-thumb]:h-5
                               [&::-webkit-slider-thumb]:rounded-full
                               [&::-webkit-slider-thumb]:bg-emerald-400
                               [&::-webkit-slider-thumb]:cursor-pointer
                               [&::-webkit-slider-thumb]:shadow-lg"
                  />
                  
                  {/* Preset buttons */}
                  <div className="flex flex-wrap gap-2 mt-4">
                    {presetDistances.map(dist => (
                      <button
                        key={dist}
                        onClick={() => {
                          setLocalDistance(dist);
                          setHoleDistance(dist);
                          playSound('click');
                        }}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all
                                  ${Math.abs(localDistance - dist) < 0.1
                                    ? 'bg-emerald-500 text-white'
                                    : 'bg-white/10 text-white/70 hover:bg-white/20'}`}
                      >
                        {dist}m
                      </button>
                    ))}
                  </div>
                </section>
                
                {/* Display Settings */}
                <section>
                  <div className="flex items-center gap-2 mb-4">
                    <Eye size={18} className="text-purple-400" />
                    <h3 className="text-sm font-semibold text-white/80 uppercase tracking-wider">
                      Display
                    </h3>
                  </div>

                  <div className="flex items-center justify-between mb-4">
                    <span className="text-white/70">Show Distance Ruler</span>
                    <button
                      onClick={() => {
                        setShowDistance(!showDistance);
                        playSound('click');
                      }}
                      className={`w-12 h-6 rounded-full transition-colors relative
                                ${showDistance ? 'bg-emerald-500' : 'bg-white/20'}`}
                    >
                      <motion.div
                        animate={{ x: showDistance ? 24 : 0 }}
                        className="absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-md"
                      />
                    </button>
                  </div>
                </section>

                {/* Sound Settings */}
                <section>
                  <div className="flex items-center gap-2 mb-4">
                    {soundSettings.enabled ? (
                      <Volume2 size={18} className="text-blue-400" />
                    ) : (
                      <VolumeX size={18} className="text-white/40" />
                    )}
                    <h3 className="text-sm font-semibold text-white/80 uppercase tracking-wider">
                      Sound
                    </h3>
                  </div>
                  
                  {/* Toggle */}
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-white/70">Sound Effects</span>
                    <button
                      onClick={() => {
                        toggleSound();
                      }}
                      className={`w-12 h-6 rounded-full transition-colors relative
                                ${soundSettings.enabled ? 'bg-emerald-500' : 'bg-white/20'}`}
                    >
                      <motion.div
                        animate={{ x: soundSettings.enabled ? 24 : 0 }}
                        className="absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-md"
                      />
                    </button>
                  </div>
                  
                  {/* Volume slider */}
                  {soundSettings.enabled && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-white/50">Volume</span>
                        <span className="text-white/70">{Math.round(soundSettings.volume * 100)}%</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.1"
                        value={soundSettings.volume}
                        onChange={(e) => setVolume(parseFloat(e.target.value))}
                        className="w-full h-2 bg-white/10 rounded-full appearance-none cursor-pointer
                                   [&::-webkit-slider-thumb]:appearance-none
                                   [&::-webkit-slider-thumb]:w-4
                                   [&::-webkit-slider-thumb]:h-4
                                   [&::-webkit-slider-thumb]:rounded-full
                                   [&::-webkit-slider-thumb]:bg-blue-400
                                   [&::-webkit-slider-thumb]:cursor-pointer"
                      />
                    </div>
                  )}
                </section>
                
                {/* Green Speed */}
                <section>
                  <div className="flex items-center gap-2 mb-4">
                    <Gauge size={18} className="text-cyan-400" />
                    <h3 className="text-sm font-semibold text-white/80 uppercase tracking-wider">
                      Green Speed
                    </h3>
                  </div>
                  
                  <p className="text-xs text-white/40 mb-3">
                    Adjust to match your putting surface
                  </p>
                  
                  <div className="space-y-2">
                    {GREEN_SPEED_PRESETS.map(preset => (
                      <button
                        key={preset.id}
                        onClick={() => handleGreenSpeedChange(preset.id)}
                        className={`w-full flex items-center justify-between p-3 rounded-xl transition-all
                                  ${greenSpeed === preset.id
                                    ? 'bg-cyan-500/20 border border-cyan-400/50'
                                    : 'bg-white/5 border border-white/10 hover:bg-white/10'}`}
                      >
                        <div className="text-left">
                          <div className={`font-medium ${greenSpeed === preset.id ? 'text-cyan-400' : 'text-white/80'}`}>
                            {preset.label}
                          </div>
                          <div className="text-xs text-white/40">{preset.description}</div>
                        </div>
                        {greenSpeed === preset.id && (
                          <div className="w-2 h-2 rounded-full bg-cyan-400" />
                        )}
                      </button>
                    ))}
                  </div>
                </section>
                
                {/* Session */}
                <section>
                  <div className="flex items-center gap-2 mb-4">
                    <RotateCcw size={18} className="text-orange-400" />
                    <h3 className="text-sm font-semibold text-white/80 uppercase tracking-wider">
                      Session
                    </h3>
                  </div>
                  
                  <button
                    onClick={handleResetSession}
                    className="w-full py-3 px-4 rounded-xl bg-orange-500/20 border border-orange-500/30
                             text-orange-400 font-medium hover:bg-orange-500/30 transition-colors"
                  >
                    Reset Session
                  </button>
                  <p className="text-xs text-white/40 mt-2">
                    Clears all statistics and starts a fresh session.
                  </p>
                </section>
                
                {/* Keyboard shortcuts */}
                <section className="pt-4 border-t border-white/10">
                  <h3 className="text-xs font-semibold text-white/40 uppercase tracking-wider mb-3">
                    Keyboard Shortcuts
                  </h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-white/50">Toggle Lab Mode</span>
                      <kbd className="px-2 py-0.5 bg-white/10 rounded text-white/70">L</kbd>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-white/50">Test Shot</span>
                      <kbd className="px-2 py-0.5 bg-white/10 rounded text-white/70">T</kbd>
                    </div>
                  </div>
                </section>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
};
