import React, { useState } from 'react';
import { Green3D } from './Green3D';
import { StatsHUD } from './StatsHUD';
import { ShotFeedback } from './ShotFeedback';
import { StudioBackground } from './StudioBackground';
import { NeuralFeedback } from './NeuralFeedback';
import { ReadyIndicator } from './ReadyIndicator';
import { DrillMode } from './DrillMode';
import { ConsistencyPanel } from './ConsistencyPanel';
import { UserSelector } from '../shared/UserSelector';
import { DistanceControl } from './DistanceControl';
import { ShotHistory } from './ShotHistory';
import { CameraSelector } from './CameraSelector';
import type { CameraView } from './CameraSelector';
import { usePuttingState } from '../../contexts/WebSocketContext';
import { Users, History, BarChart3, Settings, Target } from 'lucide-react';

export const PlayMode: React.FC = () => {
  const { sessionData, users } = usePuttingState();
  const [showConsistency, setShowConsistency] = useState(false);
  const [showUserSelector, setShowUserSelector] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showDrills, setShowDrills] = useState(false);
  const [cameraView, setCameraView] = useState<CameraView>('standard');

  const currentUserId = sessionData?.user_id;
  const currentUser = users.find(u => u.id === currentUserId);

  return (
    <div className="relative w-full h-full">
      <StudioBackground />
      <Green3D cameraView={cameraView} />
      
      {/* Main HUD (Top Center) */}
      <StatsHUD />
      
      {/* Feedback Overlays */}
      <NeuralFeedback />
      <ShotFeedback />
      <ReadyIndicator />
      
      {/* LEFT SIDEBAR - Controls & Panels */}
      <div className="absolute top-0 left-0 bottom-0 p-4 z-40 flex flex-col pointer-events-none w-96">
        <div className="mt-12 flex flex-col gap-4 pointer-events-auto max-h-full">
          {/* Top Controls Group */}
          <div className="flex flex-col gap-3 shrink-0">
            {/* User Switcher */}
            <button
              onClick={() => setShowUserSelector(true)}
              className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/80 backdrop-blur-md border border-slate-200 
                       hover:bg-white transition-all duration-200 text-sm group w-fit shadow-sm"
              title="Switch User"
            >
              <Users size={18} className="text-slate-600 group-hover:text-slate-900" />
              <span className="text-slate-700 group-hover:text-slate-900 font-medium">{currentUser ? currentUser.name : 'Guest'}</span>
            </button>
            
            {/* Mode Toggles */}
            <div className="flex flex-col gap-3 items-start">
              <button
                onClick={() => setShowHistory(!showHistory)}
                className={`p-3 rounded-full backdrop-blur-md border transition-all duration-200 shadow-sm ${
                  showHistory 
                    ? 'bg-emerald-500 text-white border-emerald-400 shadow-[0_0_15px_rgba(16,185,129,0.4)]' 
                    : 'bg-white/80 text-slate-600 border-slate-200 hover:bg-white hover:text-slate-900'
                }`}
                title="Shot History"
              >
                <History size={20} />
              </button>

              <button
                onClick={() => setShowDrills(!showDrills)}
                className={`p-3 rounded-full backdrop-blur-md border transition-all duration-200 shadow-sm ${
                  showDrills 
                    ? 'bg-blue-500 text-white border-blue-400 shadow-[0_0_15px_rgba(59,130,246,0.4)]' 
                    : 'bg-white/80 text-slate-600 border-slate-200 hover:bg-white hover:text-slate-900'
                }`}
                title="Drills"
              >
                <Target size={20} />
              </button>

              <button
                onClick={() => setShowConsistency(!showConsistency)}
                className={`p-3 rounded-full backdrop-blur-md border transition-all duration-200 shadow-sm ${
                  showConsistency 
                    ? 'bg-emerald-500 text-white border-emerald-400 shadow-[0_0_15px_rgba(16,185,129,0.4)]' 
                    : 'bg-white/80 text-slate-600 border-slate-200 hover:bg-white hover:text-slate-900'
                }`}
                title="Stats & Consistency"
              >
                <BarChart3 size={20} />
              </button>
            </div>
          </div>

          {/* Stacked Panels */}
          <div className="flex flex-col gap-4 min-h-0 overflow-y-auto pr-2 -mr-2 custom-scrollbar">
            {showHistory && (
              <ShotHistory 
                className="w-full shrink-0 max-h-[300px]" 
                onClose={() => setShowHistory(false)} 
              />
            )}

            {showConsistency && (
              <ConsistencyPanel 
                className="w-full shrink-0" 
              />
            )}

            {showDrills && (
              <div className="w-full shrink-0">
                <DrillMode />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* RIGHT SIDEBAR - Camera & Distance */}
      <div className="absolute top-0 right-0 bottom-0 p-4 z-40 flex flex-col justify-between pointer-events-none">
        {/* Top Right: Camera Controls */}
        <div className="mt-16 flex flex-col gap-4 pointer-events-auto items-end">
          <CameraSelector 
            currentView={cameraView} 
            onViewChange={setCameraView} 
          />
        </div>

        {/* Bottom Right: Distance Control */}
        <div className="pointer-events-auto">
          <DistanceControl className="w-64" />
        </div>
      </div>

      {/* Overlays */}
      {showUserSelector && (
        <div className="absolute inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center pointer-events-auto">
          <UserSelector onClose={() => setShowUserSelector(false)} className="w-96" />
        </div>
      )}
    </div>
  );
};
