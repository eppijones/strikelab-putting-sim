/**
 * Sound effects hook for putting simulator.
 * Uses Web Audio API to generate sounds programmatically.
 */

import React, { useCallback, useRef, useEffect, useState, createContext, useContext, type ReactNode } from 'react';

export interface SoundSettings {
  enabled: boolean;
  volume: number; // 0-1
}

// Generate a simple tone
const createTone = (
  context: AudioContext,
  frequency: number,
  duration: number,
  type: OscillatorType = 'sine',
  volume: number = 0.3
): { start: () => void } => {
  return {
    start: () => {
      const oscillator = context.createOscillator();
      const gainNode = context.createGain();
      
      oscillator.connect(gainNode);
      gainNode.connect(context.destination);
      
      oscillator.type = type;
      oscillator.frequency.setValueAtTime(frequency, context.currentTime);
      
      // Envelope
      gainNode.gain.setValueAtTime(0, context.currentTime);
      gainNode.gain.linearRampToValueAtTime(volume, context.currentTime + 0.01);
      gainNode.gain.exponentialRampToValueAtTime(0.001, context.currentTime + duration);
      
      oscillator.start(context.currentTime);
      oscillator.stop(context.currentTime + duration);
    }
  };
};

// Create a "ball drop in cup" sound - pleasant ding
const playHoleSound = (context: AudioContext, volume: number) => {
  // Play a pleasant chord
  const notes = [523.25, 659.25, 783.99]; // C5, E5, G5
  notes.forEach((freq, i) => {
    setTimeout(() => {
      createTone(context, freq, 0.4, 'sine', volume * 0.4).start();
    }, i * 50);
  });
  
  // Add a "plunk" sound
  setTimeout(() => {
    const osc = context.createOscillator();
    const gain = context.createGain();
    osc.connect(gain);
    gain.connect(context.destination);
    
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(200, context.currentTime);
    osc.frequency.exponentialRampToValueAtTime(80, context.currentTime + 0.15);
    
    gain.gain.setValueAtTime(volume * 0.5, context.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, context.currentTime + 0.2);
    
    osc.start(context.currentTime);
    osc.stop(context.currentTime + 0.2);
  }, 100);
};

// Create a "miss" sound - subtle thud
const playMissSound = (context: AudioContext, volume: number) => {
  const osc = context.createOscillator();
  const gain = context.createGain();
  osc.connect(gain);
  gain.connect(context.destination);
  
  osc.type = 'sine';
  osc.frequency.setValueAtTime(150, context.currentTime);
  osc.frequency.exponentialRampToValueAtTime(50, context.currentTime + 0.1);
  
  gain.gain.setValueAtTime(volume * 0.2, context.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, context.currentTime + 0.15);
  
  osc.start(context.currentTime);
  osc.stop(context.currentTime + 0.15);
};

// Create a "rolling" sound
const playRollingSound = (context: AudioContext, volume: number, speed: number): () => void => {
  const bufferSize = 2 * context.sampleRate;
  const noiseBuffer = context.createBuffer(1, bufferSize, context.sampleRate);
  const output = noiseBuffer.getChannelData(0);
  
  for (let i = 0; i < bufferSize; i++) {
    output[i] = Math.random() * 2 - 1;
  }
  
  const noise = context.createBufferSource();
  noise.buffer = noiseBuffer;
  noise.loop = true;
  
  const filter = context.createBiquadFilter();
  filter.type = 'lowpass';
  filter.frequency.setValueAtTime(100 + speed * 200, context.currentTime);
  
  const gain = context.createGain();
  gain.gain.setValueAtTime(volume * 0.05 * Math.min(speed, 1), context.currentTime);
  
  noise.connect(filter);
  filter.connect(gain);
  gain.connect(context.destination);
  
  noise.start();
  
  return () => {
    gain.gain.exponentialRampToValueAtTime(0.001, context.currentTime + 0.1);
    setTimeout(() => noise.stop(), 100);
  };
};

// Create a "lip out" sound - close but no cigar
const playLipOutSound = (context: AudioContext, volume: number) => {
  // Rising then falling tone
  const osc = context.createOscillator();
  const gain = context.createGain();
  osc.connect(gain);
  gain.connect(context.destination);
  
  osc.type = 'sine';
  osc.frequency.setValueAtTime(400, context.currentTime);
  osc.frequency.linearRampToValueAtTime(600, context.currentTime + 0.1);
  osc.frequency.linearRampToValueAtTime(300, context.currentTime + 0.3);
  
  gain.gain.setValueAtTime(volume * 0.3, context.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, context.currentTime + 0.3);
  
  osc.start(context.currentTime);
  osc.stop(context.currentTime + 0.3);
};

// Create a "streak" celebration sound
const playStreakSound = (context: AudioContext, volume: number, streakCount: number) => {
  const baseFreq = 523.25; // C5
  const notes = [0, 4, 7, 12].map(semitone => baseFreq * Math.pow(2, semitone / 12));
  
  notes.forEach((freq, i) => {
    setTimeout(() => {
      createTone(context, freq, 0.2, 'sine', volume * 0.3).start();
    }, i * 80);
  });
};

export type SoundType = 'hole' | 'miss' | 'lipOut' | 'rolling' | 'streak' | 'click';

export const useSoundEffects = () => {
  const [settings, setSettings] = useState<SoundSettings>(() => {
    // Load from localStorage
    const saved = localStorage.getItem('soundSettings');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch {
        return { enabled: true, volume: 0.5 };
      }
    }
    return { enabled: true, volume: 0.5 };
  });
  
  const audioContextRef = useRef<AudioContext | null>(null);
  const rollingStopRef = useRef<(() => void) | null>(null);
  
  // Initialize audio context on first user interaction
  const initContext = useCallback(() => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext();
    }
    if (audioContextRef.current.state === 'suspended') {
      audioContextRef.current.resume();
    }
    return audioContextRef.current;
  }, []);
  
  // Save settings to localStorage
  useEffect(() => {
    localStorage.setItem('soundSettings', JSON.stringify(settings));
  }, [settings]);
  
  const playSound = useCallback((type: SoundType, options?: { speed?: number; streakCount?: number }) => {
    if (!settings.enabled) return;
    
    const context = initContext();
    const volume = settings.volume;
    
    switch (type) {
      case 'hole':
        playHoleSound(context, volume);
        break;
      case 'miss':
        playMissSound(context, volume);
        break;
      case 'lipOut':
        playLipOutSound(context, volume);
        break;
      case 'rolling':
        if (rollingStopRef.current) {
          rollingStopRef.current();
        }
        rollingStopRef.current = playRollingSound(context, volume, options?.speed || 0.5);
        break;
      case 'streak':
        playStreakSound(context, volume, options?.streakCount || 3);
        break;
      case 'click':
        createTone(context, 800, 0.05, 'sine', volume * 0.2).start();
        break;
    }
  }, [settings, initContext]);
  
  const stopRolling = useCallback(() => {
    if (rollingStopRef.current) {
      rollingStopRef.current();
      rollingStopRef.current = null;
    }
  }, []);
  
  const toggleSound = useCallback(() => {
    setSettings(prev => ({ ...prev, enabled: !prev.enabled }));
  }, []);
  
  const setVolume = useCallback((volume: number) => {
    setSettings(prev => ({ ...prev, volume: Math.max(0, Math.min(1, volume)) }));
  }, []);
  
  return {
    settings,
    playSound,
    stopRolling,
    toggleSound,
    setVolume,
    setSettings,
  };
};

// Create a context for global sound access
interface SoundContextType {
  settings: SoundSettings;
  playSound: (type: SoundType, options?: { speed?: number; streakCount?: number }) => void;
  stopRolling: () => void;
  toggleSound: () => void;
  setVolume: (volume: number) => void;
}

const SoundContext = createContext<SoundContextType | null>(null);

export const SoundProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const soundEffects = useSoundEffects();
  
  return (
    <SoundContext.Provider value={soundEffects}>
      {children}
    </SoundContext.Provider>
  );
};

export const useSound = () => {
  const context = useContext(SoundContext);
  if (!context) {
    throw new Error('useSound must be used within a SoundProvider');
  }
  return context;
};
