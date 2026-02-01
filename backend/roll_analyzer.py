"""
Roll quality analyzer for StrikeLab Putting Sim.
Detects skid vs true roll by analyzing deceleration patterns.

Theory:
- A ball in pure roll follows predictable exponential deceleration: v(t) = v0 * e^(-kt)
- A skidding ball decelerates faster initially due to sliding friction
- The transition from skid to roll creates a characteristic "knee" in the velocity curve

This module analyzes early trajectory data to estimate:
- Skid distance (before true roll begins)
- Roll percentage (% of putt that's rolling vs skidding)
- Roll quality rating
"""

import numpy as np
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass
from scipy import optimize

logger = logging.getLogger(__name__)


@dataclass
class RollAnalysis:
    """Result of roll quality analysis."""
    # Skid metrics
    skid_distance_m: float = 0.0        # Estimated distance before true roll
    skid_percentage: float = 0.0        # % of putt that was skid
    roll_percentage: float = 100.0      # % of putt that was rolling
    
    # Roll quality
    roll_quality: str = "unknown"       # "excellent", "good", "fair", "poor"
    roll_quality_score: float = 0.0     # 0-100 score
    
    # Fit quality
    fit_r_squared: float = 0.0          # R² of deceleration fit
    confidence: float = 0.0             # Confidence in analysis (0-1)
    
    # Raw metrics
    initial_speed_m_s: float = 0.0
    final_speed_m_s: float = 0.0
    speed_drop_percentage: float = 0.0  # % speed lost in first phase
    
    # Analysis metadata
    num_samples: int = 0
    analysis_method: str = "none"


class RollAnalyzer:
    """
    Analyzes ball roll quality by examining velocity deceleration patterns.
    
    Uses curve fitting to detect the transition from skid to true roll:
    - Pure roll: v(t) = v0 * e^(-k_roll * t)
    - Skid: v(t) = v0 * e^(-k_skid * t), where k_skid > k_roll
    
    The "knee" where deceleration rate changes indicates skid-to-roll transition.
    """
    
    # Physical constants
    SKID_FRICTION_MULTIPLIER = 1.5  # Skid friction is ~1.5x roll friction
    MIN_SAMPLES_FOR_ANALYSIS = 8    # Minimum trajectory points needed
    EXCELLENT_SKID_PCT = 10.0       # <10% skid = excellent
    GOOD_SKID_PCT = 15.0            # <15% skid = good
    FAIR_SKID_PCT = 25.0            # <25% skid = fair
    
    def __init__(
        self,
        expected_roll_friction: float = 0.15,  # From predictor
        min_speed_threshold_m_s: float = 0.1,
        pixels_per_meter: float = 1150.0
    ):
        self.expected_roll_friction = expected_roll_friction
        self.min_speed_threshold = min_speed_threshold_m_s
        self.ppm = pixels_per_meter
    
    def set_calibration(self, pixels_per_meter: float):
        """Update pixels per meter calibration."""
        self.ppm = pixels_per_meter
    
    def analyze_trajectory(
        self,
        trajectory: List[Tuple[float, float, float]],  # [(x, y, timestamp_s), ...]
        total_distance_m: float
    ) -> RollAnalysis:
        """
        Analyze ball trajectory to determine roll quality.
        
        Args:
            trajectory: List of (x, y, timestamp) tuples in pixels
            total_distance_m: Total distance traveled (including virtual)
        
        Returns:
            RollAnalysis with skid detection results
        """
        result = RollAnalysis()
        
        if len(trajectory) < self.MIN_SAMPLES_FOR_ANALYSIS:
            result.analysis_method = "insufficient_data"
            return result
        
        result.num_samples = len(trajectory)
        
        # Convert trajectory to velocities
        velocities = self._compute_velocities(trajectory)
        
        if len(velocities) < 5:
            result.analysis_method = "insufficient_velocities"
            return result
        
        # Extract time and speed arrays
        times = np.array([v[0] for v in velocities])
        speeds_px_s = np.array([v[1] for v in velocities])
        speeds_m_s = speeds_px_s / self.ppm
        
        # Filter out very low speeds (noise at end)
        valid_mask = speeds_m_s > self.min_speed_threshold
        if np.sum(valid_mask) < 5:
            result.analysis_method = "speeds_too_low"
            return result
        
        times = times[valid_mask]
        speeds_m_s = speeds_m_s[valid_mask]
        
        # Normalize time to start at 0
        times = times - times[0]
        
        result.initial_speed_m_s = float(speeds_m_s[0])
        result.final_speed_m_s = float(speeds_m_s[-1])
        
        # Try different analysis methods
        try:
            # Method 1: Single exponential fit (assumes pure roll)
            pure_roll_result = self._fit_pure_roll(times, speeds_m_s)
            
            # Method 2: Piecewise fit (skid + roll)
            piecewise_result = self._fit_piecewise(times, speeds_m_s)
            
            # Compare fits - if piecewise is significantly better, skid detected
            if piecewise_result and pure_roll_result:
                # Use AIC or simple R² comparison
                if piecewise_result['r_squared'] > pure_roll_result['r_squared'] + 0.05:
                    # Piecewise fit is better - skid detected
                    result = self._build_result_from_piecewise(
                        piecewise_result, total_distance_m, len(velocities)
                    )
                    result.analysis_method = "piecewise_fit"
                else:
                    # Pure roll fit is good enough
                    result = self._build_result_from_pure_roll(
                        pure_roll_result, total_distance_m, len(velocities)
                    )
                    result.analysis_method = "pure_roll_fit"
            elif pure_roll_result:
                result = self._build_result_from_pure_roll(
                    pure_roll_result, total_distance_m, len(velocities)
                )
                result.analysis_method = "pure_roll_fit"
            else:
                # Method 3: Simple heuristic based on early deceleration
                result = self._analyze_simple_heuristic(times, speeds_m_s, total_distance_m)
                result.analysis_method = "heuristic"
                
        except Exception as e:
            logger.warning(f"Roll analysis failed: {e}")
            result.analysis_method = f"error: {str(e)}"
        
        return result
    
    def _compute_velocities(
        self,
        trajectory: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float]]:
        """Compute velocities from trajectory points."""
        velocities = []
        
        for i in range(1, len(trajectory)):
            x1, y1, t1 = trajectory[i - 1]
            x2, y2, t2 = trajectory[i]
            
            dt = t2 - t1
            if dt <= 0 or dt > 0.5:  # Skip invalid or too-long gaps
                continue
            
            dx = x2 - x1
            dy = y2 - y1
            distance = np.sqrt(dx**2 + dy**2)
            speed = distance / dt
            
            # Use midpoint time
            t_mid = (t1 + t2) / 2
            velocities.append((t_mid, speed))
        
        return velocities
    
    def _fit_pure_roll(
        self,
        times: np.ndarray,
        speeds: np.ndarray
    ) -> Optional[dict]:
        """Fit single exponential decay (pure roll model)."""
        try:
            # v(t) = v0 * exp(-k * t)
            # log(v) = log(v0) - k * t
            log_speeds = np.log(speeds)
            
            # Linear regression on log
            coeffs = np.polyfit(times, log_speeds, 1)
            k = -coeffs[0]
            v0 = np.exp(coeffs[1])
            
            # Calculate R²
            fitted = v0 * np.exp(-k * times)
            ss_res = np.sum((speeds - fitted) ** 2)
            ss_tot = np.sum((speeds - np.mean(speeds)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'v0': v0,
                'k': k,
                'r_squared': max(0, r_squared),
                'fitted': fitted
            }
        except Exception as e:
            logger.debug(f"Pure roll fit failed: {e}")
            return None
    
    def _fit_piecewise(
        self,
        times: np.ndarray,
        speeds: np.ndarray
    ) -> Optional[dict]:
        """Fit piecewise exponential (skid + roll model)."""
        try:
            n = len(times)
            if n < 8:
                return None
            
            best_result = None
            best_r_squared = -np.inf
            
            # Try different transition points
            for i in range(3, n - 3):
                t_transition = times[i]
                
                # Fit skid phase (0 to transition)
                t_skid = times[:i+1]
                s_skid = speeds[:i+1]
                
                if len(t_skid) < 3 or np.min(s_skid) <= 0:
                    continue
                
                log_s_skid = np.log(s_skid)
                coeffs_skid = np.polyfit(t_skid, log_s_skid, 1)
                k_skid = -coeffs_skid[0]
                v0_skid = np.exp(coeffs_skid[1])
                
                # Fit roll phase (transition to end)
                t_roll = times[i:] - t_transition
                s_roll = speeds[i:]
                
                if len(t_roll) < 3 or np.min(s_roll) <= 0:
                    continue
                
                log_s_roll = np.log(s_roll)
                coeffs_roll = np.polyfit(t_roll, log_s_roll, 1)
                k_roll = -coeffs_roll[0]
                v0_roll = np.exp(coeffs_roll[1])
                
                # Check physical plausibility
                # Skid friction should be higher than roll friction
                if k_skid <= k_roll * 1.1:
                    continue
                
                # Calculate combined R²
                fitted_skid = v0_skid * np.exp(-k_skid * t_skid)
                fitted_roll = v0_roll * np.exp(-k_roll * t_roll)
                
                fitted = np.concatenate([fitted_skid[:-1], fitted_roll])
                if len(fitted) != len(speeds):
                    # Adjust if needed
                    fitted = np.concatenate([fitted_skid[:i], fitted_roll])
                
                if len(fitted) != len(speeds):
                    continue
                
                ss_res = np.sum((speeds - fitted) ** 2)
                ss_tot = np.sum((speeds - np.mean(speeds)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                if r_squared > best_r_squared:
                    best_r_squared = r_squared
                    best_result = {
                        'v0_skid': v0_skid,
                        'k_skid': k_skid,
                        'v0_roll': v0_roll,
                        'k_roll': k_roll,
                        't_transition': t_transition,
                        'transition_idx': i,
                        'r_squared': max(0, r_squared),
                        'fitted': fitted
                    }
            
            return best_result
            
        except Exception as e:
            logger.debug(f"Piecewise fit failed: {e}")
            return None
    
    def _build_result_from_piecewise(
        self,
        fit: dict,
        total_distance_m: float,
        num_samples: int
    ) -> RollAnalysis:
        """Build RollAnalysis from piecewise fit results."""
        result = RollAnalysis()
        
        # Estimate skid distance from transition time and initial speed
        t_trans = fit['t_transition']
        v0 = fit['v0_skid']
        k_skid = fit['k_skid']
        
        # Distance during skid: integral of v0 * exp(-k*t) from 0 to t_trans
        # = v0/k * (1 - exp(-k * t_trans))
        skid_distance = (v0 / k_skid) * (1 - np.exp(-k_skid * t_trans))
        skid_distance_m = skid_distance / self.ppm
        
        result.skid_distance_m = float(skid_distance_m)
        result.skid_percentage = float((skid_distance_m / total_distance_m) * 100) if total_distance_m > 0 else 0
        result.roll_percentage = 100.0 - result.skid_percentage
        
        result.fit_r_squared = float(fit['r_squared'])
        result.confidence = min(1.0, fit['r_squared'])
        result.num_samples = num_samples
        
        # Speed drop during skid phase
        v_at_transition = v0 * np.exp(-k_skid * t_trans)
        result.speed_drop_percentage = float((1 - v_at_transition / v0) * 100)
        
        # Determine roll quality
        result.roll_quality, result.roll_quality_score = self._rate_roll_quality(result.skid_percentage)
        
        return result
    
    def _build_result_from_pure_roll(
        self,
        fit: dict,
        total_distance_m: float,
        num_samples: int
    ) -> RollAnalysis:
        """Build RollAnalysis from pure roll fit results."""
        result = RollAnalysis()
        
        # Pure roll means minimal skid
        result.skid_distance_m = 0.0
        result.skid_percentage = 0.0
        result.roll_percentage = 100.0
        
        result.fit_r_squared = float(fit['r_squared'])
        result.confidence = min(1.0, fit['r_squared'])
        result.num_samples = num_samples
        result.initial_speed_m_s = float(fit['v0'] / self.ppm)
        result.speed_drop_percentage = 0.0
        
        # Excellent roll quality for pure roll
        result.roll_quality = "excellent"
        result.roll_quality_score = 95.0
        
        return result
    
    def _analyze_simple_heuristic(
        self,
        times: np.ndarray,
        speeds: np.ndarray,
        total_distance_m: float
    ) -> RollAnalysis:
        """Simple heuristic analysis when curve fitting fails."""
        result = RollAnalysis()
        
        # Look at deceleration rate in first few samples vs later samples
        n = len(speeds)
        if n < 6:
            return result
        
        # Split into early (first 30%) and late (last 30%)
        split_early = max(2, n // 3)
        split_late = min(n - 2, 2 * n // 3)
        
        early_speeds = speeds[:split_early]
        late_speeds = speeds[split_late:]
        early_times = times[:split_early]
        late_times = times[split_late:]
        
        # Compute average deceleration rates
        if len(early_speeds) >= 2 and len(late_speeds) >= 2:
            early_decel = (early_speeds[0] - early_speeds[-1]) / (early_times[-1] - early_times[0])
            late_decel = (late_speeds[0] - late_speeds[-1]) / (late_times[-1] - late_times[0])
            
            # If early deceleration is significantly higher, likely skidding
            if early_decel > 0 and late_decel > 0:
                decel_ratio = early_decel / late_decel
                
                if decel_ratio > 1.3:
                    # Estimate skid percentage based on decel ratio
                    estimated_skid_pct = min(30, (decel_ratio - 1) * 20)
                    result.skid_percentage = float(estimated_skid_pct)
                    result.roll_percentage = 100.0 - result.skid_percentage
                    result.skid_distance_m = float(total_distance_m * estimated_skid_pct / 100)
                else:
                    result.skid_percentage = 5.0  # Assume minimal skid
                    result.roll_percentage = 95.0
                    result.skid_distance_m = float(total_distance_m * 0.05)
        
        result.initial_speed_m_s = float(speeds[0])
        result.final_speed_m_s = float(speeds[-1])
        result.num_samples = n
        result.confidence = 0.5  # Lower confidence for heuristic
        
        result.roll_quality, result.roll_quality_score = self._rate_roll_quality(result.skid_percentage)
        
        return result
    
    def _rate_roll_quality(self, skid_percentage: float) -> Tuple[str, float]:
        """Rate roll quality based on skid percentage."""
        if skid_percentage < self.EXCELLENT_SKID_PCT:
            quality = "excellent"
            score = 90 + (self.EXCELLENT_SKID_PCT - skid_percentage)
        elif skid_percentage < self.GOOD_SKID_PCT:
            quality = "good"
            score = 75 + (self.GOOD_SKID_PCT - skid_percentage)
        elif skid_percentage < self.FAIR_SKID_PCT:
            quality = "fair"
            score = 50 + (self.FAIR_SKID_PCT - skid_percentage)
        else:
            quality = "poor"
            score = max(0, 50 - (skid_percentage - self.FAIR_SKID_PCT))
        
        return quality, min(100, max(0, score))


# Singleton instance
_roll_analyzer: Optional[RollAnalyzer] = None


def get_roll_analyzer() -> RollAnalyzer:
    """Get the global roll analyzer instance."""
    global _roll_analyzer
    if _roll_analyzer is None:
        _roll_analyzer = RollAnalyzer()
    return _roll_analyzer
