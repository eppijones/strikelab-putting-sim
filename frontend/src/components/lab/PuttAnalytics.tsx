import React, { useState, useEffect } from 'react';
import { usePuttingState, type ShotResult } from '../../contexts/WebSocketContext';
import { Target, Wind, Maximize, ArrowUpRight, Hash } from 'lucide-react';
import clsx from 'clsx';

interface StatCardProps {
  label: string;
  value: string | number;
  unit?: string;
  icon: React.ReactNode;
  color?: string;
  highlight?: boolean;
}

const StatCard: React.FC<StatCardProps> = ({ label, value, unit, icon, color = 'text-white', highlight = false }) => (
  <div className={clsx(
    "bg-sl-panel border rounded-lg p-3 flex flex-col gap-1 transition-all",
    highlight ? "border-sl-cyan/50 bg-sl-cyan/5" : "border-white/5"
  )}>
    <div className="flex items-center gap-2 text-xs text-gray-400 uppercase tracking-wider">
      {icon}
      <span>{label}</span>
    </div>
    <div className={clsx("text-2xl font-mono font-medium", color)}>
      {value}<span className="text-sm text-gray-500 ml-1">{unit}</span>
    </div>
  </div>
);

export const PuttAnalytics: React.FC = () => {
  const { lastJsonMessage, sessionData, gameState } = usePuttingState();
  const [displayShot, setDisplayShot] = useState<ShotResult | null>(null);
  const [isNewShot, setIsNewShot] = useState(false);

  // Update display shot whenever we get new valid shot data
  useEffect(() => {
    if (lastJsonMessage?.shot) {
      setDisplayShot(lastJsonMessage.shot);
      setIsNewShot(true);
      
      // Clear the highlight after 2 seconds
      const timer = setTimeout(() => setIsNewShot(false), 2000);
      return () => clearTimeout(timer);
    }
  }, [lastJsonMessage?.shot?.distance_m, lastJsonMessage?.shot?.speed_m_s]);

  const isMeasuring = gameState === 'TRACKING' || gameState === 'VIRTUAL_ROLLING';
  const totalPutts = sessionData?.total_putts || 0;

  // Format direction as degrees with L/R indicator
  const formatDirection = (deg: number) => {
    const absDeg = Math.abs(deg);
    const direction = deg > 0.1 ? 'R' : deg < -0.1 ? 'L' : '';
    return { value: absDeg.toFixed(1), direction };
  };

  const directionData = displayShot ? formatDirection(displayShot.direction_deg) : null;

  return (
    <div className="flex flex-col gap-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm text-gray-400 uppercase tracking-wider font-medium">Putt Analytics</h3>
        {isMeasuring && (
          <span className="text-xs text-sl-cyan animate-pulse font-medium">MEASURING...</span>
        )}
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {/* Total Putts */}
        <StatCard
          label="Total Putts"
          value={totalPutts}
          icon={<Hash size={14} />}
          color="text-sl-accent"
        />

        {/* Latest Shot Stats - only show after first putt */}
        {displayShot ? (
          <>
            <StatCard
              label="Speed"
              value={displayShot.speed_m_s.toFixed(2)}
              unit="m/s"
              icon={<Wind size={14} />}
              color="text-sl-green"
              highlight={isNewShot}
            />
            <StatCard
              label="Distance"
              value={isMeasuring ? '...' : displayShot.distance_m.toFixed(2)}
              unit={isMeasuring ? '' : 'm'}
              icon={<Maximize size={14} />}
              color="text-sl-cyan"
              highlight={isNewShot}
            />
            <StatCard
              label="Line"
              value={directionData?.value || '0.0'}
              unit={`° ${directionData?.direction || ''}`}
              icon={<ArrowUpRight size={14} />}
              color={Math.abs(displayShot.direction_deg) < 1.0 ? 'text-sl-green' : 'text-yellow-400'}
              highlight={isNewShot}
            />
          </>
        ) : (
          <>
            <StatCard
              label="Speed"
              value="—"
              unit="m/s"
              icon={<Wind size={14} />}
              color="text-gray-500"
            />
            <StatCard
              label="Distance"
              value="—"
              unit="m"
              icon={<Maximize size={14} />}
              color="text-gray-500"
            />
            <StatCard
              label="Line"
              value="—"
              unit="°"
              icon={<ArrowUpRight size={14} />}
              color="text-gray-500"
            />
          </>
        )}
      </div>

      {/* Additional info when no shots yet */}
      {!displayShot && totalPutts === 0 && (
        <div className="text-center text-gray-500 text-sm py-2">
          <Target size={16} className="inline mr-2 opacity-50" />
          Make a putt to see analytics
        </div>
      )}
    </div>
  );
};
