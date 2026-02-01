import React from 'react';
import { Green3D } from './Green3D';
import { StatsHUD } from './StatsHUD';
import { ShotFeedback } from './ShotFeedback';
import { StudioBackground } from './StudioBackground';
import { NeuralFeedback } from './NeuralFeedback';
import { ReadyIndicator } from './ReadyIndicator';
import { DrillMode } from './DrillMode';

export const PlayMode: React.FC = () => {
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
    </div>
  );
};
