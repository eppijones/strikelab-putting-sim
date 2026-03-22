export type GameState = 'ARMED' | 'TRACKING' | 'VIRTUAL_ROLLING' | 'STOPPED' | 'COOLDOWN';

export interface BallData {
  x_px: number;
  y_px: number;
  radius_px: number;
  confidence: number;
}

export interface VelocityData {
  vx_px_s: number;
  vy_px_s: number;
  speed_px_s: number;
}

export interface ShotResult {
  speed_m_s: number;
  speed_px_s: number;
  direction_deg: number;
  physical_distance_m: number;
  virtual_distance_m: number;
  distance_m: number;
  distance_px: number;
  trajectory: number[][];
  exited_frame: boolean;
}

export interface PredictionData {
  trajectory: number[][];
  final_position: [number, number];
  final_time_s: number;
  exit_speed_px_s: number;
}

export interface Metrics {
  cap_fps: number;
  proc_fps: number;
  disp_fps: number;
  proc_latency_ms: number;
  idle_stddev: number;
}

export interface VirtualBall {
  x: number;
  y: number;
  vx: number;
  vy: number;
  speed_m_s: number;
  distance_m: number;
  is_rolling: boolean;
}

export type ShotResultType =
  | 'pending'
  | 'made'
  | 'miss_short'
  | 'miss_long'
  | 'miss_left'
  | 'miss_right'
  | 'lip_out';

export interface GameStateData {
  hole: {
    distance_m: number;
    position_x_m: number;
    position_y_m: number;
    radius_m: number;
  };
  last_shot: {
    result: ShotResultType;
    distance_to_hole_m: number;
    lateral_miss_m: number;
    depth_miss_m: number;
    miss_description: string;
    is_made: boolean;
  } | null;
}

export interface ConsistencyMetrics {
  speed_stddev: number;
  direction_stddev: number;
  distance_error_stddev: number;
  speed_cv: number;
  consistency_score: number;
  rolling_speed_stddev: number;
  rolling_direction_stddev: number;
}

export interface TendencyAnalysis {
  speed_bias_m_s: number;
  distance_bias_m: number;
  direction_bias_deg: number;
  lateral_bias_m: number;
  dominant_miss: string;
  dominant_miss_percentage: number;
  speed_tendency: string;
  direction_tendency: string;
}

export interface MissDistribution {
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

export interface SessionData {
  session_id: string;
  duration_s: number;
  total_putts: number;
  putts_made: number;
  make_percentage: number;
  current_streak: number;
  best_streak: number;
  avg_speed_m_s: number;
  avg_miss_distance_m: number;
  putts_by_distance: Record<string, { total: number; made: number; percentage: number }>;
  consistency: ConsistencyMetrics;
  tendencies: TendencyAnalysis;
  miss_distribution: MissDistribution;
  user_id?: number | null;
}

export interface User {
  id: number;
  name: string;
  handicap: number;
  created_at: string;
}

export type DrillType = 'none' | 'distance_control' | 'ladder_drill';

export interface DrillData {
  active: boolean;
  drill_type: DrillType;
  current_target_m?: number;
  total_points?: number;
  attempts?: number;
  targets_completed?: number;
  duration_s?: number;
  ladder_position?: number;
  last_attempt?: {
    rating: string;
    points: number;
    error_cm: number;
  };
}

export interface ClubMetricsData {
  club_path_deg: number;
  face_angle_deg: number;
  attack_angle_deg: number;
  club_speed_m_s: number;
  stroke_tempo: number;
  backswing_time_ms: number;
  forward_swing_time_ms: number;
  backswing_length_m: number;
  forward_swing_length_m: number;
  stroke_phase: string;
  impact_point?: [number, number];
  available?: boolean;
}

export interface BallMetricsData {
  speed_m_s: number;
  distance_m: number;
  direction_deg: number;
  launch_angle_deg: number;
  skid_distance_m: number;
  true_roll_pct: number;
  carry_distance_m: number;
  roll_distance_m: number;
  peak_height_m: number;
  landing_angle_deg: number;
  spin_estimate_rpm: number;
}

export interface ShotReportData {
  shot_type: 'putt' | 'chip' | 'unknown';
  ball: BallMetricsData;
  club: ClubMetricsData;
  trajectory_2d: number[][];
  trajectory_3d: number[][];
  club_path_3d: number[][];
  result: string;
  is_made: boolean;
  cameras_used: string[];
  fast_putt_resolved: boolean;
  fast_putt_estimated: boolean;
}

export interface CameraStatusData {
  type: string;
  connected: boolean;
  running: boolean;
  fps: number;
  resolution: number[];
  error?: string;
  frame_count?: number;
  last_frame_age_ms?: number;
  target_fps?: number;
  min_healthy_fps?: number;
  driver_reported_fps?: number;
  consecutive_read_failures?: number;
}

export interface MultiCameraState {
  cameras: Record<string, CameraStatusData>;
  shot_report: ShotReportData | null;
  club: {
    stroke_phase: string;
    metrics?: ClubMetricsData;
  };
  system_health?: {
    all_streams_reporting: boolean;
    max_last_frame_age_ms?: number | null;
    stale_warning: boolean;
  };
  tracking_activity?: {
    game_state: string;
    arducam_ball_tracking: boolean;
    zed_club_phase: string;
    realsense_launch_active: boolean;
  };
}

export interface BackendState {
  frame_id?: number;
  timestamp_ms: number;
  state: GameState;
  lane: string;
  ball: BallData | null;
  ball_visible: boolean;
  velocity: VelocityData | null;
  prediction: PredictionData | null;
  virtual_ball: VirtualBall | null;
  exit_state?: {
    position: [number, number];
    velocity: [number, number];
    speed_px_s: number;
    direction_deg: number;
    curvature: number;
    trajectory_before_exit: number[][];
  } | null;
  shot: ShotResult | null;
  metrics: Metrics & Record<string, unknown>;
  calibrated: boolean;
  auto_calibrated: boolean;
  lens_calibrated: boolean;
  pixels_per_meter: number;
  calibration_source?: string;
  calibration_confidence?: number;
  overlay_radius_scale: number;
  resolution: [number, number] | number[];
  ready_status?: string;
  game?: GameStateData | null;
  session?: SessionData | null;
  drill?: DrillData | null;
  multi_camera?: MultiCameraState | null;
}
