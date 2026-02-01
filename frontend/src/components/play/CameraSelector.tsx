import React, { useState } from 'react';
import { Camera, ChevronDown, Video, Eye, Move3d, ArrowDown } from 'lucide-react';

export type CameraView = 'standard' | 'free' | 'top-down' | 'cinematic';

interface CameraSelectorProps {
  currentView: CameraView;
  onViewChange: (view: CameraView) => void;
  className?: string;
}

const viewOptions: { id: CameraView; label: string; icon: React.ReactNode; description: string }[] = [
  { 
    id: 'standard', 
    label: 'Standard', 
    icon: <Camera size={16} />, 
    description: 'Default follow camera' 
  },
  { 
    id: 'free', 
    label: 'Free Cam', 
    icon: <Move3d size={16} />, 
    description: 'Drag to orbit, scroll to zoom' 
  },
  { 
    id: 'top-down', 
    label: 'Top Down', 
    icon: <ArrowDown size={16} />, 
    description: 'Bird\'s eye view • Scroll to zoom' 
  },
  { 
    id: 'cinematic', 
    label: 'Cinematic', 
    icon: <Video size={16} />, 
    description: 'Smooth cinematic angles' 
  },
];

export const CameraSelector: React.FC<CameraSelectorProps> = ({ 
  currentView, 
  onViewChange, 
  className = '' 
}) => {
  const [isOpen, setIsOpen] = useState(false);
  
  const currentOption = viewOptions.find(v => v.id === currentView) || viewOptions[0];

  return (
    <div className={`relative ${className}`}>
      {/* Toggle Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`p-3 rounded-full backdrop-blur-md border transition-all duration-200 shadow-sm ${
          isOpen 
            ? 'bg-purple-500 text-white border-purple-400 shadow-[0_0_15px_rgba(168,85,247,0.4)]' 
            : 'bg-white/80 text-slate-600 border-slate-200 hover:bg-white hover:text-slate-900'
        }`}
        title={`Camera: ${currentOption.label}`}
      >
        {currentOption.icon}
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="absolute top-full right-0 mt-2 w-56 bg-black/80 backdrop-blur-xl rounded-xl border border-white/10 overflow-hidden shadow-2xl z-50">
          {viewOptions.map((option) => (
            <button
              key={option.id}
              onClick={() => {
                onViewChange(option.id);
                setIsOpen(false);
              }}
              className={`w-full flex items-center gap-3 px-4 py-3 text-left transition-all ${
                currentView === option.id
                  ? 'bg-purple-500/20 text-purple-400'
                  : 'text-white/80 hover:bg-white/5 hover:text-white'
              }`}
            >
              <span className={`${currentView === option.id ? 'text-purple-400' : 'text-white/60'}`}>
                {option.icon}
              </span>
              <div className="flex-1">
                <div className="text-sm font-medium">{option.label}</div>
                <div className="text-xs text-white/50">{option.description}</div>
              </div>
              {currentView === option.id && (
                <div className="w-2 h-2 rounded-full bg-purple-400" />
              )}
            </button>
          ))}
          
          {/* Hint for Free Cam or Top Down */}
          {(currentView === 'free' || currentView === 'top-down') && (
            <div className="px-4 py-2 bg-purple-500/10 border-t border-white/5">
              <p className="text-xs text-purple-300/80">
                {currentView === 'free' 
                  ? 'Drag to rotate • Scroll to zoom • Right-drag to pan'
                  : 'Scroll to zoom • Drag to pan'}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
