import React from 'react';
import { Wifi, WifiOff } from 'lucide-react';
import { usePuttingState } from '../../contexts/WebSocketContext';
import clsx from 'clsx';

export const ConnectionStatus: React.FC = () => {
  const { isConnected } = usePuttingState();

  return (
    <div className={clsx(
      "fixed top-4 left-4 z-50 flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border backdrop-blur-md transition-colors duration-500",
      isConnected 
        ? "bg-green-500/10 border-green-500/20 text-green-400" 
        : "bg-red-500/10 border-red-500/20 text-red-400"
    )}>
      {isConnected ? <Wifi size={14} /> : <WifiOff size={14} />}
      <span>{isConnected ? 'CONNECTED' : 'DISCONNECTED'}</span>
    </div>
  );
};
