import React from 'react';
import { usePuttingState } from '../../contexts/WebSocketContext';

// Local type definitions (mirrored from WebSocketContext to avoid import issues)
interface ConsistencyMetrics {
  speed_stddev: number;
  direction_stddev: number;
  distance_error_stddev: number;
  speed_cv: number;
  consistency_score: number;
  rolling_speed_stddev: number;
  rolling_direction_stddev: number;
}

interface TendencyAnalysis {
  speed_bias_m_s: number;
  distance_bias_m: number;
  direction_bias_deg: number;
  lateral_bias_m: number;
  dominant_miss: string;
  dominant_miss_percentage: number;
  speed_tendency: string;
  direction_tendency: string;
}

interface MissDistribution {
  right_short: number;
  right_long: number;
  left_short: number;
  left_long: number;
  right_short_pct: number;
  right_long_pct: number;
  left_short_pct: number;
  left_long_pct: number;
  total_right: number;
  total_left: number;
  total_short: number;
  total_long: number;
  total_misses: number;
}

interface ConsistencyPanelProps {
  className?: string;
}

// Consistency score gauge component
const ConsistencyGauge: React.FC<{ score: number; label: string }> = ({ score, label }) => {
  // Color based on score
  const getColor = (s: number) => {
    if (s >= 80) return '#22c55e'; // green
    if (s >= 60) return '#84cc16'; // lime
    if (s >= 40) return '#eab308'; // yellow
    if (s >= 20) return '#f97316'; // orange
    return '#ef4444'; // red
  };

  const color = getColor(score);
  const circumference = 2 * Math.PI * 36; // radius = 36
  const strokeDashoffset = circumference - (score / 100) * circumference;

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-20 h-20">
        <svg className="w-20 h-20 transform -rotate-90">
          {/* Background circle */}
          <circle
            cx="40"
            cy="40"
            r="36"
            stroke="rgba(255,255,255,0.1)"
            strokeWidth="6"
            fill="transparent"
          />
          {/* Progress circle */}
          <circle
            cx="40"
            cy="40"
            r="36"
            stroke={color}
            strokeWidth="6"
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className="transition-all duration-500"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-xl font-bold text-white">{Math.round(score)}</span>
        </div>
      </div>
      <span className="text-xs text-gray-400 mt-1">{label}</span>
    </div>
  );
};

// Tendency arrow indicator
const TendencyArrow: React.FC<{ direction: string; value: number; label: string }> = ({ direction, value, label }) => {
  const isNeutral = direction === 'neutral' || direction === 'none';
  const isRight = direction.includes('right') || direction.includes('pushing');
  const isLong = direction.includes('long');
  
  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="text-gray-400 w-20">{label}</span>
      <div className="flex items-center gap-1">
        {isNeutral ? (
          <div className="w-4 h-4 rounded-full bg-green-500/50 flex items-center justify-center">
            <span className="text-xs text-green-300">•</span>
          </div>
        ) : (
          <div className={`flex items-center ${isRight || isLong ? 'text-orange-400' : 'text-blue-400'}`}>
            {isRight ? (
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 4l8 8-8 8V4z"/>
              </svg>
            ) : direction.includes('left') || direction.includes('pulling') ? (
              <svg className="w-4 h-4 transform rotate-180" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 4l8 8-8 8V4z"/>
              </svg>
            ) : isLong ? (
              <svg className="w-4 h-4 transform -rotate-90" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 4l8 8-8 8V4z"/>
              </svg>
            ) : (
              <svg className="w-4 h-4 transform rotate-90" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 4l8 8-8 8V4z"/>
              </svg>
            )}
          </div>
        )}
        <span className={`text-xs ${isNeutral ? 'text-green-400' : 'text-gray-300'}`}>
          {direction === 'none' ? 'neutral' : direction}
        </span>
      </div>
    </div>
  );
};

// Miss distribution heatmap (2x2 grid)
const MissHeatmap: React.FC<{ distribution: MissDistribution }> = ({ distribution }) => {
  const getIntensity = (pct: number) => {
    if (pct >= 40) return 'bg-red-500/70';
    if (pct >= 25) return 'bg-orange-500/60';
    if (pct >= 15) return 'bg-yellow-500/50';
    if (pct >= 5) return 'bg-blue-500/40';
    return 'bg-gray-500/20';
  };

  const total = distribution.total_misses;
  if (total === 0) {
    return (
      <div className="text-center text-gray-500 text-sm py-4">
        No misses to analyze yet
      </div>
    );
  }

  return (
    <div className="relative">
      {/* Direction labels */}
      <div className="absolute -left-8 top-1/2 -translate-y-1/2 text-xs text-gray-500 transform -rotate-90">
        LEFT
      </div>
      <div className="absolute -right-8 top-1/2 -translate-y-1/2 text-xs text-gray-500 transform rotate-90">
        RIGHT
      </div>
      <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-5 text-xs text-gray-500">
        LONG
      </div>
      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-5 text-xs text-gray-500">
        SHORT
      </div>
      
      {/* 2x2 Grid */}
      <div className="grid grid-cols-2 gap-1 w-32 h-32 mx-auto">
        {/* Left Long */}
        <div className={`rounded-tl-lg ${getIntensity(distribution.left_long_pct)} flex items-center justify-center`}>
          <span className="text-white text-sm font-medium">
            {distribution.left_long_pct > 0 ? `${distribution.left_long_pct}%` : ''}
          </span>
        </div>
        {/* Right Long */}
        <div className={`rounded-tr-lg ${getIntensity(distribution.right_long_pct)} flex items-center justify-center`}>
          <span className="text-white text-sm font-medium">
            {distribution.right_long_pct > 0 ? `${distribution.right_long_pct}%` : ''}
          </span>
        </div>
        {/* Left Short */}
        <div className={`rounded-bl-lg ${getIntensity(distribution.left_short_pct)} flex items-center justify-center`}>
          <span className="text-white text-sm font-medium">
            {distribution.left_short_pct > 0 ? `${distribution.left_short_pct}%` : ''}
          </span>
        </div>
        {/* Right Short */}
        <div className={`rounded-br-lg ${getIntensity(distribution.right_short_pct)} flex items-center justify-center`}>
          <span className="text-white text-sm font-medium">
            {distribution.right_short_pct > 0 ? `${distribution.right_short_pct}%` : ''}
          </span>
        </div>
      </div>
      
      {/* Center hole indicator */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-white border-2 border-gray-800" />
    </div>
  );
};

// Stddev bar visualization
const StddevBar: React.FC<{ label: string; value: number; max: number; unit: string; benchmark?: number }> = ({ 
  label, value, max, unit, benchmark 
}) => {
  const percentage = Math.min(100, (value / max) * 100);
  const isGood = benchmark ? value <= benchmark : percentage < 50;
  
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-gray-400">{label}</span>
        <span className={isGood ? 'text-green-400' : 'text-orange-400'}>
          {value.toFixed(2)} {unit}
        </span>
      </div>
      <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
        <div 
          className={`h-full rounded-full transition-all duration-300 ${isGood ? 'bg-green-500' : 'bg-orange-500'}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      {benchmark && (
        <div className="text-xs text-gray-500 text-right">
          Tour avg: {benchmark} {unit}
        </div>
      )}
    </div>
  );
};

export const ConsistencyPanel: React.FC<ConsistencyPanelProps> = ({ className = '' }) => {
  const { sessionData, isConnected } = usePuttingState();

  if (!isConnected || !sessionData) {
    return null;
  }

  const { consistency, tendencies, miss_distribution } = sessionData;
  
  // Don't show if no data yet
  if (sessionData.total_putts < 3) {
    return (
      <div className={`bg-black/60 backdrop-blur-sm rounded-xl p-4 border border-white/10 ${className}`}>
        <h3 className="text-white font-semibold mb-2">Consistency Analysis</h3>
        <p className="text-gray-400 text-sm">Hit at least 3 putts to see consistency metrics</p>
      </div>
    );
  }

  return (
    <div className={`bg-black/60 backdrop-blur-sm rounded-xl p-4 border border-white/10 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-white font-semibold">Consistency Analysis</h3>
        <span className="text-xs text-gray-400">{sessionData.total_putts} putts</span>
      </div>

      {/* Main consistency score */}
      <div className="flex justify-center mb-4">
        <ConsistencyGauge score={consistency?.consistency_score || 0} label="Consistency Score" />
      </div>

      {/* Stddev metrics */}
      <div className="space-y-3 mb-4">
        <StddevBar 
          label="Direction σ" 
          value={consistency?.direction_stddev || 0} 
          max={5} 
          unit="°"
          benchmark={1.0}
        />
        <StddevBar 
          label="Speed σ" 
          value={consistency?.speed_stddev || 0} 
          max={0.5} 
          unit="m/s"
          benchmark={0.1}
        />
        <StddevBar 
          label="Distance Error σ" 
          value={consistency?.distance_error_stddev || 0} 
          max={1.0} 
          unit="m"
          benchmark={0.3}
        />
      </div>

      {/* Tendencies */}
      <div className="border-t border-white/10 pt-3 mb-4">
        <h4 className="text-sm text-gray-300 mb-2">Tendencies</h4>
        <div className="space-y-1">
          <TendencyArrow 
            direction={tendencies?.direction_tendency || 'neutral'} 
            value={tendencies?.direction_bias_deg || 0}
            label="Direction"
          />
          <TendencyArrow 
            direction={tendencies?.speed_tendency || 'neutral'} 
            value={tendencies?.distance_bias_m || 0}
            label="Distance"
          />
        </div>
        
        {/* Dominant miss */}
        {tendencies?.dominant_miss && tendencies.dominant_miss !== 'none' && (
          <div className="mt-2 text-xs">
            <span className="text-gray-400">Dominant miss: </span>
            <span className="text-orange-400">
              {tendencies.dominant_miss} ({tendencies.dominant_miss_percentage}%)
            </span>
          </div>
        )}
      </div>

      {/* Miss distribution heatmap */}
      <div className="border-t border-white/10 pt-3">
        <h4 className="text-sm text-gray-300 mb-3 text-center">Miss Distribution</h4>
        <MissHeatmap distribution={miss_distribution || {
          right_short: 0, right_long: 0, left_short: 0, left_long: 0,
          right_short_pct: 0, right_long_pct: 0, left_short_pct: 0, left_long_pct: 0,
          total_right: 0, total_left: 0, total_short: 0, total_long: 0,
          total_misses: 0
        }} />
      </div>
    </div>
  );
};

export default ConsistencyPanel;
