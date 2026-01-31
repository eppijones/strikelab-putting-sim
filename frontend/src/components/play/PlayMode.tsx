import React from 'react';
import { GreenVisualizer } from './GreenVisualizer';
import { StatsHUD } from './StatsHUD';
import { ShotFeedback } from './ShotFeedback';
import { AmbientBackground } from '../shared/AmbientBackground';

export const PlayMode: React.FC = () => {
  return (
    <div className="relative w-full h-full">
      <AmbientBackground />
      <GreenVisualizer />
      <StatsHUD />
      <ShotFeedback />
    </div>
  );
};
