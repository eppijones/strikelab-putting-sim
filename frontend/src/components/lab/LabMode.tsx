import React, { useState } from 'react';
import { CameraOverlay } from './CameraOverlay';
import { MetricsPanel } from './MetricsPanel';
import { DebugInfo } from './DebugInfo';
import { PuttAnalytics } from './PuttAnalytics';
import MultiCameraView from './MultiCameraView';
import { Camera, Grid3x3 } from 'lucide-react';

export const LabMode: React.FC = () => {
  const [viewMode, setViewMode] = useState<'single' | 'multi'>('multi');

  return (
    <div className="flex flex-col h-full w-full p-4 gap-4">
      {/* View Toggle */}
      <div className="flex items-center gap-2 shrink-0">
        <button
          onClick={() => setViewMode('single')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-all ${
            viewMode === 'single'
              ? 'bg-[var(--sl-cyan)]/20 text-[var(--sl-cyan)] border border-[var(--sl-cyan)]/30'
              : 'bg-white/5 text-white/40 border border-white/5 hover:text-white/60'
          }`}
        >
          <Camera size={12} /> Primary
        </button>
        <button
          onClick={() => setViewMode('multi')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-all ${
            viewMode === 'multi'
              ? 'bg-[var(--sl-cyan)]/20 text-[var(--sl-cyan)] border border-[var(--sl-cyan)]/30'
              : 'bg-white/5 text-white/40 border border-white/5 hover:text-white/60'
          }`}
        >
          <Grid3x3 size={12} /> All Cameras
        </button>
      </div>

      {/* Top Row: Camera Feed(s) */}
      <div className="flex-1 min-h-0">
        {viewMode === 'single' ? (
          <CameraOverlay />
        ) : (
          <MultiCameraView />
        )}
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
