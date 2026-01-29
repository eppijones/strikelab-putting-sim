"""
Lens distortion calibration for StrikeLab Putting Sim.

Uses OpenCV's camera calibration with a checkerboard pattern to compute
lens distortion coefficients. This is a one-time setup that corrects for
barrel distortion in the camera lens.

Usage:
    python -m backend.lens_calibration --capture    # Capture calibration images
    python -m backend.lens_calibration --calibrate  # Run calibration from saved images
    python -m backend.lens_calibration --live       # Interactive live calibration
"""

import cv2
import numpy as np
import json
import logging
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Default checkerboard size (inner corners)
DEFAULT_CHECKERBOARD = (9, 6)

# Calibration data storage
CALIBRATION_DIR = Path("calibration")
LENS_PARAMS_FILE = CALIBRATION_DIR / "lens_params.json"
CALIBRATION_IMAGES_DIR = CALIBRATION_DIR / "images"


@dataclass
class LensCalibrationResult:
    """Result of lens calibration."""
    success: bool
    camera_matrix: Optional[List[List[float]]] = None  # 3x3 intrinsic matrix
    dist_coeffs: Optional[List[float]] = None          # Distortion coefficients [k1, k2, p1, p2, k3]
    image_size: Optional[Tuple[int, int]] = None       # (width, height)
    reprojection_error: float = 0.0                    # RMS reprojection error in pixels
    num_images_used: int = 0
    error_message: Optional[str] = None
    calibrated_at: Optional[str] = None


class LensCalibrator:
    """
    Handles camera lens distortion calibration using checkerboard pattern.
    
    Workflow:
    1. Capture 15-20 images of checkerboard at different angles/positions
    2. Detect checkerboard corners in each image
    3. Compute camera matrix and distortion coefficients
    4. Save calibration to JSON for use in main application
    """
    
    def __init__(
        self,
        checkerboard_size: Tuple[int, int] = DEFAULT_CHECKERBOARD,
        square_size_mm: float = 25.0
    ):
        """
        Args:
            checkerboard_size: Number of inner corners (columns, rows)
            square_size_mm: Physical size of each square in millimeters
        """
        self.checkerboard_size = checkerboard_size
        self.square_size_mm = square_size_mm
        
        # Prepare object points (3D points in real world space)
        # These are the same for all calibration images
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size_mm  # Scale to real-world units
        
        # Storage for calibration data
        self.obj_points: List[np.ndarray] = []  # 3D points in real world
        self.img_points: List[np.ndarray] = []  # 2D points in image plane
        self.image_size: Optional[Tuple[int, int]] = None
        
        # Calibration results
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.reprojection_error: float = 0.0
        
        # Create directories if needed
        CALIBRATION_DIR.mkdir(exist_ok=True)
        CALIBRATION_IMAGES_DIR.mkdir(exist_ok=True)
    
    def detect_checkerboard(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect checkerboard corners in a frame.
        
        Args:
            frame: BGR image
            
        Returns:
            (found, corners) - found is True if checkerboard detected, corners are the corner points
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        found, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, flags)
        
        if found:
            # Refine corner locations to subpixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        return found, corners
    
    def add_calibration_frame(self, frame: np.ndarray) -> bool:
        """
        Add a frame to the calibration set if checkerboard is detected.
        
        Args:
            frame: BGR image
            
        Returns:
            True if checkerboard was found and added
        """
        found, corners = self.detect_checkerboard(frame)
        
        if found:
            self.obj_points.append(self.objp.copy())
            self.img_points.append(corners)
            
            if self.image_size is None:
                h, w = frame.shape[:2]
                self.image_size = (w, h)
            
            logger.info(f"Added calibration frame {len(self.img_points)}")
            return True
        
        return False
    
    def calibrate(self) -> LensCalibrationResult:
        """
        Run camera calibration using collected images.
        
        Returns:
            LensCalibrationResult with calibration data
        """
        if len(self.obj_points) < 10:
            return LensCalibrationResult(
                success=False,
                error_message=f"Need at least 10 images, have {len(self.obj_points)}"
            )
        
        if self.image_size is None:
            return LensCalibrationResult(
                success=False,
                error_message="No image size recorded"
            )
        
        logger.info(f"Running calibration with {len(self.obj_points)} images...")
        
        try:
            # Run calibration
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                self.obj_points,
                self.img_points,
                self.image_size,
                None,
                None
            )
            
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            self.reprojection_error = ret
            
            # Calculate reprojection error per image
            total_error = 0
            for i in range(len(self.obj_points)):
                imgpoints2, _ = cv2.projectPoints(
                    self.obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
                )
                error = cv2.norm(self.img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                total_error += error
            
            mean_error = total_error / len(self.obj_points)
            
            logger.info(f"Calibration successful!")
            logger.info(f"  RMS reprojection error: {ret:.4f} pixels")
            logger.info(f"  Mean reprojection error: {mean_error:.4f} pixels")
            logger.info(f"  Camera matrix:\n{camera_matrix}")
            logger.info(f"  Distortion coefficients: {dist_coeffs.flatten()}")
            
            from datetime import datetime, timezone
            
            return LensCalibrationResult(
                success=True,
                camera_matrix=camera_matrix.tolist(),
                dist_coeffs=dist_coeffs.flatten().tolist(),
                image_size=self.image_size,
                reprojection_error=ret,
                num_images_used=len(self.obj_points),
                calibrated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return LensCalibrationResult(
                success=False,
                error_message=str(e)
            )
    
    def save_calibration(self, result: LensCalibrationResult, path: Optional[Path] = None) -> bool:
        """Save calibration result to JSON file."""
        if not result.success:
            logger.error("Cannot save failed calibration")
            return False
        
        save_path = path or LENS_PARAMS_FILE
        
        try:
            data = asdict(result)
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved lens calibration to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return False
    
    def draw_checkerboard(self, frame: np.ndarray, corners: Optional[np.ndarray] = None) -> np.ndarray:
        """Draw detected checkerboard corners on frame."""
        frame_copy = frame.copy()
        
        if corners is None:
            found, corners = self.detect_checkerboard(frame)
            if not found:
                return frame_copy
        
        cv2.drawChessboardCorners(frame_copy, self.checkerboard_size, corners, True)
        return frame_copy
    
    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Undistort a frame using calibration data.
        
        Args:
            frame: Distorted BGR image
            
        Returns:
            Undistorted image
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            return frame
        
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)


def load_lens_calibration(path: Optional[Path] = None) -> Optional[LensCalibrationResult]:
    """
    Load lens calibration from JSON file.
    
    Returns:
        LensCalibrationResult if file exists and is valid, None otherwise
    """
    load_path = path or LENS_PARAMS_FILE
    
    if not load_path.exists():
        logger.info(f"No lens calibration file found at {load_path}")
        return None
    
    try:
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        result = LensCalibrationResult(
            success=data.get("success", False),
            camera_matrix=data.get("camera_matrix"),
            dist_coeffs=data.get("dist_coeffs"),
            image_size=tuple(data["image_size"]) if data.get("image_size") else None,
            reprojection_error=data.get("reprojection_error", 0.0),
            num_images_used=data.get("num_images_used", 0),
            calibrated_at=data.get("calibrated_at")
        )
        
        if result.success and result.camera_matrix and result.dist_coeffs:
            logger.info(f"Loaded lens calibration from {load_path}")
            logger.info(f"  Reprojection error: {result.reprojection_error:.4f} px")
            logger.info(f"  Images used: {result.num_images_used}")
            return result
        else:
            logger.warning("Lens calibration file exists but is invalid")
            return None
            
    except Exception as e:
        logger.error(f"Failed to load lens calibration: {e}")
        return None


def get_undistort_maps(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute undistortion maps for fast undistortion.
    
    Use these maps with cv2.remap() for faster undistortion than cv2.undistort().
    
    Args:
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        image_size: (width, height)
        
    Returns:
        (map1, map2) for use with cv2.remap()
    """
    # Get optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, image_size, 1, image_size
    )
    
    # Compute undistortion maps
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, image_size, cv2.CV_16SC2
    )
    
    return map1, map2


def run_live_calibration(
    camera_id: int = 0,
    checkerboard_size: Tuple[int, int] = DEFAULT_CHECKERBOARD,
    num_images: int = 15
):
    """
    Interactive live calibration using webcam/camera.
    
    Press SPACE to capture when checkerboard is detected.
    Press 'c' to run calibration after capturing enough images.
    Press 'q' to quit.
    """
    print(f"\n=== Lens Calibration Tool ===")
    print(f"Checkerboard size: {checkerboard_size[0]}x{checkerboard_size[1]} inner corners")
    print(f"Target images: {num_images}")
    print()
    print("Instructions:")
    print("  1. Hold checkerboard pattern in front of camera")
    print("  2. Move to different positions and angles")
    print("  3. Press SPACE when green corners appear to capture")
    print("  4. Press 'c' after capturing 15+ images to calibrate")
    print("  5. Press 'q' to quit")
    print()
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    # Set camera resolution (adjust as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    
    calibrator = LensCalibrator(checkerboard_size=checkerboard_size)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Detect checkerboard
        found, corners = calibrator.detect_checkerboard(frame)
        
        # Draw visualization
        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, checkerboard_size, corners, True)
            status = f"Checkerboard FOUND - Press SPACE to capture ({len(calibrator.img_points)}/{num_images})"
            color = (0, 255, 0)
        else:
            status = f"Searching for checkerboard... ({len(calibrator.img_points)}/{num_images})"
            color = (0, 0, 255)
        
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if len(calibrator.img_points) >= num_images:
            cv2.putText(display, "Press 'c' to calibrate", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Lens Calibration", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') and found:
            # Capture this frame
            if calibrator.add_calibration_frame(frame):
                print(f"Captured image {len(calibrator.img_points)}")
                # Save image for reference
                img_path = CALIBRATION_IMAGES_DIR / f"calib_{len(calibrator.img_points):02d}.jpg"
                cv2.imwrite(str(img_path), frame)
        
        elif key == ord('c') and len(calibrator.img_points) >= 10:
            # Run calibration
            print("\nRunning calibration...")
            result = calibrator.calibrate()
            
            if result.success:
                print(f"\nCalibration successful!")
                print(f"RMS reprojection error: {result.reprojection_error:.4f} pixels")
                
                # Save calibration
                calibrator.save_calibration(result)
                print(f"\nCalibration saved to {LENS_PARAMS_FILE}")
                
                # Show undistorted preview
                print("\nShowing undistorted preview (press any key to exit)...")
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    undistorted = calibrator.undistort_frame(frame)
                    
                    # Show side by side
                    combined = np.hstack([frame, undistorted])
                    h, w = combined.shape[:2]
                    combined = cv2.resize(combined, (w // 2, h // 2))
                    cv2.putText(combined, "Original", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(combined, "Undistorted", (combined.shape[1] // 2 + 10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow("Lens Calibration", combined)
                    if cv2.waitKey(1) & 0xFF != 255:
                        break
                break
            else:
                print(f"Calibration failed: {result.error_message}")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main entry point for lens calibration tool."""
    parser = argparse.ArgumentParser(description="Lens Distortion Calibration Tool")
    parser.add_argument("--live", action="store_true", help="Run interactive live calibration")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--cols", type=int, default=9, help="Checkerboard columns (inner corners)")
    parser.add_argument("--rows", type=int, default=6, help="Checkerboard rows (inner corners)")
    parser.add_argument("--num-images", type=int, default=15, help="Number of calibration images to capture")
    parser.add_argument("--verify", action="store_true", help="Verify existing calibration")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.verify:
        # Load and verify existing calibration
        result = load_lens_calibration()
        if result:
            print(f"\nLens calibration found:")
            print(f"  Reprojection error: {result.reprojection_error:.4f} pixels")
            print(f"  Images used: {result.num_images_used}")
            print(f"  Image size: {result.image_size}")
            print(f"  Calibrated at: {result.calibrated_at}")
        else:
            print("\nNo valid lens calibration found.")
            print("Run with --live to create calibration.")
    
    elif args.live:
        run_live_calibration(
            camera_id=args.camera,
            checkerboard_size=(args.cols, args.rows),
            num_images=args.num_images
        )
    
    else:
        parser.print_help()
        print("\n\nExamples:")
        print("  python -m backend.lens_calibration --live        # Interactive calibration")
        print("  python -m backend.lens_calibration --verify      # Check existing calibration")
        print("  python -m backend.lens_calibration --live --camera 1  # Use camera 1")


if __name__ == "__main__":
    main()
