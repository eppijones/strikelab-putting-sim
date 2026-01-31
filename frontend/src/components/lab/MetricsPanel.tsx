import React from 'react';
import { usePuttingState } from '../../contexts/WebSocketContext';
import { Activity, Clock, Gauge, Zap } from 'lucide-react';
import clsx from 'clsx';

interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  icon: React.ReactNode;
  color?: string;
}

const MetricCard: React.FC<MetricCardProps> = ({ label, value, unit, icon, color = 'text-white' }) => (
  <div className="bg-sl-panel border border-white/5 rounded-lg p-3 flex flex-col gap-1">
    <div className="flex items-center gap-2 text-xs text-gray-400 uppercase tracking-wider">
      {icon}
      <span>{label}</span>
    </div>
    <div className={clsx("text-2xl font-mono font-medium", color)}>
      {value}<span className="text-sm text-gray-500 ml-1">{unit}</span>
    </div>
  </div>
);

export const MetricsPanel: React.FC = () => {
  const { lastJsonMessage } = usePuttingState();
  const metrics = lastJsonMessage?.metrics;

  if (!metrics) return null;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 w-full">
      <MetricCard 
        label="Capture FPS" 
        value={metrics.cap_fps.toFixed(1)} 
        icon={<Activity size={14} />} 
        color={metrics.cap_fps < 30 ? 'text-red-400' : 'text-sl-green'}
      />
      <MetricCard 
        label="Process FPS" 
        value={metrics.proc_fps.toFixed(1)} 
        icon={<Zap size={14} />} 
        color={metrics.proc_fps < 30 ? 'text-yellow-400' : 'text-sl-cyan'}
      />
      <MetricCard 
        label="Latency" 
        value={metrics.proc_latency_ms.toFixed(1)} 
        unit="ms"
        icon={<Clock size={14} />} 
        color={metrics.proc_latency_ms > 20 ? 'text-orange-400' : 'text-white'}
      />
      <MetricCard 
        label="Idle Jitter" 
        value={metrics.idle_stddev.toFixed(2)} 
        unit="px"
        icon={<Gauge size={14} />} 
        color={metrics.idle_stddev > 0.5 ? 'text-red-400' : 'text-gray-300'}
      />
    </div>
  );
};
