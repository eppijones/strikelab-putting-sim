import React from 'react';
import { Wifi, WifiOff, FlaskConical } from 'lucide-react';
import { usePuttingState } from '../../contexts/WebSocketContext';
import clsx from 'clsx';

export const ConnectionStatus: React.FC = () => {
  const { isConnected, isTestShotActive, triggerTestShot } = usePuttingState();

  return (
    <div className="fixed top-4 left-4 z-50 flex items-center gap-2">
      {/* Connection Status */}
      <div className={clsx(
        "flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border backdrop-blur-md transition-colors duration-500",
        isConnected 
          ? "bg-green-500/10 border-green-500/20 text-green-400" 
          : "bg-red-500/10 border-red-500/20 text-red-400"
      )}>
        {isConnected ? <Wifi size={14} /> : <WifiOff size={14} />}
        <span>{isConnected ? 'CONNECTED' : 'DISCONNECTED'}</span>
      </div>

      {/* Test Shot Button */}
      <button
        onClick={triggerTestShot}
        className={clsx(
          "flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border backdrop-blur-md transition-all duration-300",
          isTestShotActive
            ? "bg-amber-500/20 border-amber-500/40 text-amber-300 animate-pulse"
            : "bg-white/5 border-white/10 text-white/60 hover:bg-white/10 hover:text-white/80"
        )}
        title="Press T for test shot"
      >
        <FlaskConical size={14} />
        <span className="hidden sm:inline">{isTestShotActive ? 'TEST ACTIVE' : 'Test'}</span>
        <kbd className="hidden sm:inline px-1.5 py-0.5 bg-white/10 rounded text-[10px] font-mono">T</kbd>
      </button>
    </div>
  );
};
