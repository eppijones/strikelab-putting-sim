import React, { useState } from 'react';
import { Green3D } from './Green3D';
import { StatsHUD } from './StatsHUD';
import { ShotFeedback } from './ShotFeedback';
import { StudioBackground } from './StudioBackground';
import { NeuralFeedback } from './NeuralFeedback';
import { ReadyIndicator } from './ReadyIndicator';
import { DrillMode } from './DrillMode';
import { ConsistencyPanel } from './ConsistencyPanel';

export const PlayMode: React.FC = () => {
  const [showConsistency, setShowConsistency] = useState(true);

  return (
    <div className="relative w-full h-full">
      <StudioBackground />
      <Green3D />
      <StatsHUD />
      <NeuralFeedback />
      <ShotFeedback />
      <ReadyIndicator />
      {/* Drill panel - bottom left */}
      <DrillMode className="absolute bottom-4 left-4 w-80" />
      
      {/* Consistency panel - bottom right, toggleable */}
      {showConsistency && (
        <ConsistencyPanel className="absolute bottom-4 right-4 w-72" />
      )}
      
      {/* Toggle button for consistency panel */}
      <button
        onClick={() => setShowConsistency(!showConsistency)}
        className="absolute bottom-4 right-4 transform translate-x-[-18rem] bg-black/40 hover:bg-black/60 text-white/60 hover:text-white p-2 rounded-lg transition-all"
        title={showConsistency ? "Hide analytics" : "Show analytics"}
        style={{ display: showConsistency ? 'none' : 'block' }}
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      </button>
    </div>
  );
};
