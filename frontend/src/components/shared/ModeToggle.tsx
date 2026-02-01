import React, { useEffect } from 'react';
import { Beaker, Gamepad2 } from 'lucide-react';
import clsx from 'clsx';

export type AppMode = 'PLAY' | 'LAB';

interface ModeToggleProps {
  mode: AppMode;
  setMode: (mode: AppMode) => void;
}

export const ModeToggle: React.FC<ModeToggleProps> = ({ mode, setMode }) => {
  // Keyboard shortcut 'L' to toggle
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === 'l' && !e.repeat && !e.ctrlKey && !e.metaKey && !e.altKey && document.activeElement?.tagName !== 'INPUT') {
        setMode(mode === 'PLAY' ? 'LAB' : 'PLAY');
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [mode, setMode]);

  return (
    <div className="flex items-center gap-2 bg-white/80 backdrop-blur-md rounded-full p-1 border border-slate-200 shadow-sm">
      <button
        onClick={() => setMode('PLAY')}
        className={clsx(
          "flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all duration-300",
          mode === 'PLAY' 
            ? "bg-sl-green text-sl-dark shadow-md" 
            : "text-slate-500 hover:text-slate-900 hover:bg-black/5"
        )}
      >
        <Gamepad2 size={16} />
        <span className="hidden sm:inline">PLAY</span>
      </button>

      <button
        onClick={() => setMode('LAB')}
        className={clsx(
          "flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all duration-300",
          mode === 'LAB' 
            ? "bg-sl-cyan text-sl-dark shadow-md" 
            : "text-slate-500 hover:text-slate-900 hover:bg-black/5"
        )}
      >
        <Beaker size={16} />
        <span className="hidden sm:inline">LAB</span>
      </button>
    </div>
  );
};
