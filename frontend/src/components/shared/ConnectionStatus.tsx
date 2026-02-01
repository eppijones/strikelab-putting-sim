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
        "flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border backdrop-blur-md transition-colors duration-500 shadow-sm",
        isConnected 
          ? "bg-green-100 border-green-200 text-green-700" 
          : "bg-red-100 border-red-200 text-red-700"
      )}>
        {isConnected ? <Wifi size={14} /> : <WifiOff size={14} />}
        <span>{isConnected ? 'CONNECTED' : 'DISCONNECTED'}</span>
      </div>

      {/* Test Shot Button */}
      <button
        onClick={triggerTestShot}
        className={clsx(
          "flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border backdrop-blur-md transition-all duration-300 shadow-sm",
          isTestShotActive
            ? "bg-amber-100 border-amber-300 text-amber-700 animate-pulse"
            : "bg-white/80 border-slate-200 text-slate-600 hover:bg-white hover:text-slate-900"
        )}
        title="Press T for test shot"
      >
        <FlaskConical size={14} />
        <span className="hidden sm:inline">{isTestShotActive ? 'TEST ACTIVE' : 'Test'}</span>
        <kbd className="hidden sm:inline px-1.5 py-0.5 bg-black/5 rounded text-[10px] font-mono border border-black/10">T</kbd>
      </button>
    </div>
  );
};
