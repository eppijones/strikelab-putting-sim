import React from 'react';
import { Green3D } from './Green3D';
import { StatsHUD } from './StatsHUD';
import { ShotFeedback } from './ShotFeedback';
import { StudioBackground } from './StudioBackground';
import { NeuralFeedback } from './NeuralFeedback';

export const PlayMode: React.FC = () => {
  return (
    <div className="relative w-full h-full">
      <StudioBackground />
      <Green3D />
      <StatsHUD />
      <NeuralFeedback />
      <ShotFeedback />
    </div>
  );
};
