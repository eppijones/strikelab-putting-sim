import React from 'react';
import { usePuttingState } from '../../contexts/WebSocketContext';
import { RotateCcw } from 'lucide-react';

export const DebugInfo: React.FC = () => {
  const { lastJsonMessage, pixelsPerMeter, sendReset } = usePuttingState();

  if (!lastJsonMessage) return null;

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Control Bar */}
      <div className="flex items-center justify-between bg-sl-panel border border-white/5 rounded-lg p-3">
        <div className="flex flex-col">
          <span className="text-xs text-gray-400 uppercase tracking-wider">Calibration</span>
          <div className="flex items-baseline gap-2">
            <span className="text-xl font-mono text-sl-accent">{pixelsPerMeter.toFixed(1)}</span>
            <span className="text-sm text-gray-500">px/m</span>
          </div>
        </div>
        
        <div className="flex gap-2">
          <div className="flex flex-col items-end px-3 border-r border-white/10">
            <span className="text-xs text-gray-400">Lens</span>
            <span className={lastJsonMessage.lens_calibrated ? "text-sl-green" : "text-red-500"}>
              {lastJsonMessage.lens_calibrated ? "Calibrated" : "Raw"}
            </span>
          </div>
          <div className="flex flex-col items-end px-3">
            <span className="text-xs text-gray-400">Auto-Cal</span>
            <span className={lastJsonMessage.auto_calibrated ? "text-sl-green" : "text-gray-500"}>
              {lastJsonMessage.auto_calibrated ? "Active" : "Pending"}
            </span>
          </div>
        </div>

        <button 
          onClick={sendReset}
          className="flex items-center gap-2 px-3 py-2 bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/30 rounded transition-colors"
        >
          <RotateCcw size={14} />
          <span className="text-sm font-medium">RESET TRACKER</span>
        </button>
      </div>

      {/* Raw JSON View */}
      <div className="flex-1 bg-black/50 border border-white/5 rounded-lg p-4 font-mono text-xs overflow-auto text-gray-300">
        <pre>{JSON.stringify(lastJsonMessage, null, 2)}</pre>
      </div>
    </div>
  );
};
