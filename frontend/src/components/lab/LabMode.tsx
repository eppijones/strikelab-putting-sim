import React from 'react';
import { CameraOverlay } from './CameraOverlay';
import { MetricsPanel } from './MetricsPanel';
import { DebugInfo } from './DebugInfo';
import { PuttAnalytics } from './PuttAnalytics';

export const LabMode: React.FC = () => {
  return (
    <div className="flex flex-col h-full w-full p-4 gap-4">
      {/* Top Row: Camera Feed */}
      <div className="flex-1 min-h-0">
        <CameraOverlay />
      </div>

      {/* Bottom Row: Metrics, Analytics & Debug */}
      <div className="h-1/3 flex gap-4 min-h-[300px]">
        <div className="flex-1 flex flex-col gap-4">
          <MetricsPanel />
          <PuttAnalytics />
          <div className="flex-1 min-h-0">
             <DebugInfo />
          </div>
        </div>
      </div>
    </div>
  );
};
