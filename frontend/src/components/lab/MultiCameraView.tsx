import React, { useRef, useEffect, useState, useCallback } from 'react';
import { usePuttingState } from '../../contexts/WebSocketContext';
import { apiUrl } from '../../config/backend';
import { Video, VideoOff, Activity, RefreshCw } from 'lucide-react';

/**
 * Self-healing MJPEG <img> that auto-reconnects when the stream drops.
 * Detects both onError (connection refused) and stalls (stream stops
 * sending frames but TCP stays open) via a periodic liveness check.
 */
const ReconnectingMjpeg: React.FC<{
  src: string;
  alt: string;
  className?: string;
  style?: React.CSSProperties;
}> = ({ src, alt, className, style }) => {
  const imgRef = useRef<HTMLImageElement>(null);
  const [reconnecting, setReconnecting] = useState(false);
  const [activeSrc, setActiveSrc] = useState(src);
  const lastLoadedAt = useRef(Date.now());
  const retryCount = useRef(0);

  const reconnect = useCallback(() => {
    retryCount.current += 1;
    setReconnecting(true);
    const sep = src.includes('?') ? '&' : '?';
    setActiveSrc(`${src}${sep}_t=${Date.now()}`);
  }, [src]);

  useEffect(() => {
    setActiveSrc(src);
    lastLoadedAt.current = Date.now();
    retryCount.current = 0;
  }, [src]);

  useEffect(() => {
    const interval = setInterval(() => {
      const img = imgRef.current;
      if (!img) return;

      if (img.naturalWidth > 0) {
        lastLoadedAt.current = Date.now();
        if (reconnecting) setReconnecting(false);
        retryCount.current = 0;
        return;
      }

      const staleSec = (Date.now() - lastLoadedAt.current) / 1000;
      if (staleSec > 4) {
        reconnect();
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [reconnect, reconnecting]);

  return (
    <div className="relative" style={style}>
      <img
        ref={imgRef}
        src={activeSrc}
        alt={alt}
        className={className}
        style={{ minHeight: style?.minHeight, maxHeight: style?.maxHeight, width: '100%', height: '100%' }}
        onError={() => {
          setTimeout(reconnect, 1000 + Math.min(retryCount.current * 500, 3000));
        }}
        onLoad={() => {
          lastLoadedAt.current = Date.now();
          if (reconnecting) setReconnecting(false);
          retryCount.current = 0;
        }}
      />
      {reconnecting && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/60">
          <div className="flex items-center gap-2 text-white/60 text-xs">
            <RefreshCw size={14} className="animate-spin" />
            <span>Reconnecting...</span>
          </div>
        </div>
      )}
    </div>
  );
};

interface CameraFeedProps {
  label: string;
  streamUrl: string;
  status?: { connected: boolean; running: boolean; fps: number; resolution: number[] };
  accent: string;
  roleLabel: string;
  activityLabel: string;
  activityTone: 'ok' | 'warn' | 'alert' | 'idle';
  frameAgeMs?: number;
  detailLine?: string;
}

const CameraFeed: React.FC<CameraFeedProps> = ({
  label,
  streamUrl,
  status,
  accent,
  roleLabel,
  activityLabel,
  activityTone,
  frameAgeMs,
  detailLine,
}) => {
  const isActive = status?.connected && status?.running;
  const activityToneClass =
    activityTone === 'ok' ? 'bg-green-500/20 text-green-300 border-green-500/30' :
    activityTone === 'warn' ? 'bg-amber-500/20 text-amber-300 border-amber-500/30' :
    activityTone === 'alert' ? 'bg-red-500/20 text-red-300 border-red-500/30 animate-pulse' :
    'bg-white/10 text-white/50 border-white/15';

  return (
    <div className="relative rounded-xl overflow-hidden border border-white/10 bg-black/40">
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between px-3 py-1.5 bg-gradient-to-b from-black/70 to-transparent">
        <div className="flex items-center gap-2">
          <div className={`w-1.5 h-1.5 rounded-full ${isActive ? accent : 'bg-red-500'}`} />
          <span className="text-[11px] font-medium text-white/80 uppercase tracking-wider">{label}</span>
        </div>
        {status && isActive && (
          <div className="flex items-center gap-2 text-[10px] text-white/40">
            <span>{status.fps.toFixed(0)} fps</span>
            <span>{status.resolution[0]}×{status.resolution[1]}</span>
            {typeof frameAgeMs === 'number' && <span>{frameAgeMs.toFixed(0)}ms age</span>}
          </div>
        )}
      </div>

      {/* Video feed */}
      {isActive ? (
        <ReconnectingMjpeg
          src={streamUrl}
          alt={label}
          className="w-full h-full object-contain"
          style={{ minHeight: '180px', maxHeight: '240px' }}
        />
      ) : (
        <div className="flex flex-col items-center justify-center gap-2 py-12 text-white/20">
          <VideoOff size={24} />
          <span className="text-xs">Not connected</span>
        </div>
      )}

      {/* Role + activity badges */}
      <div className="absolute bottom-2 left-2 right-2 z-10 flex items-end justify-between gap-2">
        <span className="text-[10px] uppercase tracking-wider bg-black/50 border border-white/10 rounded px-2 py-1 text-white/70">
          {roleLabel}
        </span>
        <div className="flex flex-col items-end gap-1">
          <span className={`text-[10px] uppercase tracking-wider rounded border px-2 py-1 ${activityToneClass}`}>
            {activityLabel}
          </span>
          {detailLine && <span className="text-[10px] text-white/45 bg-black/40 rounded px-1.5 py-0.5">{detailLine}</span>}
        </div>
      </div>
    </div>
  );
};

const MultiCameraView: React.FC = () => {
  const { multiCamera, lastJsonMessage } = usePuttingState();

  const arducamStatus = multiCamera?.cameras?.arducam;
  const zedStatus = multiCamera?.cameras?.zed;
  const rsStatus = multiCamera?.cameras?.realsense;

  const trackingActivity = multiCamera?.tracking_activity;
  const systemHealth = multiCamera?.system_health;
  const clubPhase = multiCamera?.club?.stroke_phase || 'idle';
  const clubMetrics = multiCamera?.club?.metrics;
  const isShotActive = (trackingActivity?.game_state || '').toLowerCase() === 'tracking'
    || (trackingActivity?.game_state || '').toLowerCase() === 'virtual_rolling';

  const arducamFps = arducamStatus?.fps ?? lastJsonMessage?.metrics?.cap_fps ?? 0;
  const arducamMinHealthy = arducamStatus?.min_healthy_fps ?? 90;
  const arducamFrameAge = arducamStatus?.last_frame_age_ms;
  const arducamTracking = Boolean(trackingActivity?.arducam_ball_tracking);
  const arducamTone: CameraFeedProps['activityTone'] =
    !arducamStatus?.running ? 'alert' :
    arducamTracking ? 'ok' :
    arducamFps < arducamMinHealthy ? 'warn' : 'idle';
  const arducamLabel = arducamTracking
    ? 'Ball Tracking'
    : arducamFps < arducamMinHealthy
      ? 'Low FPS'
      : 'Armed';

  const zedFrameAge = zedStatus?.last_frame_age_ms;
  const zedFresh = typeof zedFrameAge === 'number' ? zedFrameAge < 250 : false;
  const zedTone: CameraFeedProps['activityTone'] =
    !zedStatus?.running ? 'alert' :
    clubPhase !== 'idle' ? 'ok' :
    zedFresh ? 'idle' : 'warn';
  const zedLabel = !zedStatus?.running
    ? 'Offline'
    : clubPhase !== 'idle'
      ? `Club ${clubPhase}`
      : zedFresh
        ? 'Depth Ready'
        : 'Stale Frames';

  const rsFrameAge = rsStatus?.last_frame_age_ms;
  const rsFresh = typeof rsFrameAge === 'number' ? rsFrameAge < 250 : false;
  const rsLaunchActive = Boolean(trackingActivity?.realsense_launch_active);
  const rsTone: CameraFeedProps['activityTone'] =
    !rsStatus?.running ? 'alert' :
    rsLaunchActive ? 'ok' :
    (isShotActive && !rsFresh) ? 'warn' : 'idle';
  const rsLabel = !rsStatus?.running
    ? 'Offline'
    : rsLaunchActive
      ? 'Launch Tracking'
      : rsFresh
        ? 'Launch Ready'
        : 'Stale Frames';

  return (
    <div className="space-y-3">
      {/* Camera Grid */}
      <div className="grid grid-cols-3 gap-3">
        {/* Arducam (primary) */}
        <CameraFeed
          label="Arducam OV9281"
          streamUrl={apiUrl('/api/video')}
          status={{
            connected: arducamStatus?.connected ?? (lastJsonMessage !== null),
            running: arducamStatus?.running ?? (lastJsonMessage !== null),
            fps: arducamFps,
            resolution: lastJsonMessage?.resolution ? Array.from(lastJsonMessage.resolution) : [0, 0],
          }}
          accent="bg-green-400"
          roleLabel="Primary Ball Tracker"
          activityLabel={arducamLabel}
          activityTone={arducamTone}
          frameAgeMs={arducamFrameAge}
          detailLine={`target ${Math.round(arducamStatus?.target_fps ?? 120)} fps`}
        />

        {/* ZED 2i */}
        <CameraFeed
          label="ZED 2i Depth"
          streamUrl={apiUrl('/api/video/zed')}
          status={zedStatus}
          accent="bg-blue-400"
          roleLabel="3D Ball + Club"
          activityLabel={zedLabel}
          activityTone={zedTone}
          frameAgeMs={zedFrameAge}
          detailLine={zedStatus?.running ? `${Math.round(zedStatus?.fps ?? 0)} fps stream` : undefined}
        />

        {/* RealSense D455 */}
        <CameraFeed
          label="RealSense D455"
          streamUrl={apiUrl('/api/video/realsense')}
          status={rsStatus}
          accent="bg-purple-400"
          roleLabel="Launch / Chip Sensor"
          activityLabel={rsLabel}
          activityTone={rsTone}
          frameAgeMs={rsFrameAge}
          detailLine={rsStatus?.running ? `${Math.round(rsStatus?.fps ?? 0)} fps stream` : undefined}
        />
      </div>

      {systemHealth && (
        <div className={`rounded-xl border px-3 py-2 text-xs ${
          systemHealth.stale_warning
            ? 'border-amber-500/40 bg-amber-500/10 text-amber-200'
            : 'border-emerald-500/30 bg-emerald-500/10 text-emerald-200'
        }`}>
          {systemHealth.stale_warning
            ? 'Camera sync warning: one or more streams are stale.'
            : 'All active camera streams are reporting healthy frame cadence.'}
        </div>
      )}

      {/* Club Tracker Status */}
      {(zedStatus?.connected) && (
        <div className="rounded-xl border border-white/10 bg-[var(--sl-panel)]/50 p-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity size={14} className="text-[var(--sl-cyan)]" />
              <span className="text-xs font-medium text-white/60 uppercase tracking-wider">Club Tracker</span>
            </div>
            <span className={`text-[10px] px-2 py-0.5 rounded-full ${
              clubPhase === 'idle' ? 'bg-white/5 text-white/30' :
              clubPhase === 'impact' ? 'bg-red-500/20 text-red-400' :
              'bg-[var(--sl-cyan)]/20 text-[var(--sl-cyan)]'
            }`}>
              {clubPhase.toUpperCase()}
            </span>
          </div>

          {clubMetrics && (
            <div className="mt-2 grid grid-cols-4 gap-2 text-center">
              <div>
                <div className="text-[10px] text-white/30">Path</div>
                <div className="text-sm text-white/80">{clubMetrics.club_path_deg.toFixed(1)}°</div>
              </div>
              <div>
                <div className="text-[10px] text-white/30">Face</div>
                <div className="text-sm text-white/80">{clubMetrics.face_angle_deg.toFixed(1)}°</div>
              </div>
              <div>
                <div className="text-[10px] text-white/30">Speed</div>
                <div className="text-sm text-white/80">{clubMetrics.club_speed_m_s.toFixed(1)}</div>
              </div>
              <div>
                <div className="text-[10px] text-white/30">Tempo</div>
                <div className="text-sm text-white/80">{clubMetrics.stroke_tempo.toFixed(1)}</div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default MultiCameraView;
