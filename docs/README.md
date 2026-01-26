# StrikeLab - Putting Sim (LAB v1)

Real-time putting launch monitor using a top-down camera to capture ball speed and direction.

## Quick Start

### Prerequisites

- Python 3.11+
- Arducam OV9281 camera (or any UVC camera for testing)
- Good lighting (avoid flickering)

### Installation

```bash
cd StrikeLab_PuttingSim
pip install -r requirements.txt
```

### Running

**With Arducam OV9281:**
```bash
python -m backend.main --arducam
```

**With standard webcam (testing):**
```bash
python -m backend.main --webcam
```

**With video file replay:**
```bash
python -m backend.main --replay path/to/video.mp4
```

**Options:**
```
--host HOST     Server host (default: 0.0.0.0)
--port PORT     Server port (default: 8000)
--debug         Enable debug logging
```

### Access the UI

Open http://localhost:8000 in your browser.

## Features

- **Real-time ball tracking** at up to 120fps
- **Shot detection** with state machine (ARMED → TRACKING → STOPPED)
- **Speed and direction** measurement in calibrated world coordinates
- **Performance metrics**: CAP/PROC/DISP FPS, latency, jitter
- **Manual calibration** via 4-point rectangle

## Acceptance Metrics

| Metric | Target |
|--------|--------|
| Idle jitter | < 2px stddev over 5 seconds |
| Impact latency | ≤ 2 frames to TRACKING state |
| Speed availability | ≤ 5 frames to first speed estimate |
| FPS truth | Displayed CAP/PROC/DISP use real timestamps |

## Calibration

1. Click "Calibrate" button in the UI
2. Click 4 corners of a known rectangle (clockwise from top-left)
3. Enter the real-world dimensions in meters
4. Click "Apply"

The calibration defines:
- World coordinate system origin
- +X axis direction ("forward toward hole")
- Pixels-per-meter scale factor

Calibration is saved to `config.json`.

## Troubleshooting

### Camera not detected

**Linux:**
```bash
# List video devices
v4l2-ctl --list-devices

# Check supported formats
v4l2-ctl -d /dev/video0 --list-formats-ext
```

**macOS:**
- Check System Preferences → Security & Privacy → Camera
- Try different `device_id` values (0, 1, 2...)

### Low FPS

1. **Reduce resolution** in config (try 640x480)
2. **Check USB bandwidth** - use USB 3.0 port
3. **Disable other cameras**
4. **Check exposure** - faster exposure = higher FPS possible

### Ball not detected

1. **Lighting**: Ensure even, bright lighting
2. **Ball color**: White ball on green background works best
3. **Threshold tuning**: Adjust HSV thresholds in config:
   ```json
   "detector": {
     "white_lower_v": 180,  // Lower = detect darker whites
     "white_upper_s": 60    // Higher = allow more saturation
   }
   ```

### Flickering detection

- Avoid fluorescent lights (60Hz flicker)
- Use DC-powered LED lighting
- Increase exposure time (reduces flicker sensitivity)

## Architecture

```
backend/
  main.py        - FastAPI server + WebSocket + capture loop
  camera.py      - Camera abstraction (Arducam/replay/webcam)
  detector.py    - Ball detection (HSV thresholding)
  tracker.py     - Two-lane tracker + state machine
  calibration.py - Homography for world coordinates
  predictor.py   - Ball trajectory prediction
  config.py      - Configuration management

frontend/
  index.html     - Main page
  style.css      - Styling
  app.js         - Canvas renderer + WebSocket client
```

## WebSocket Message Schema

```json
{
  "frame_id": 12345,
  "timestamp_ms": 1706198400123.456,
  "state": "TRACKING",
  "lane": "MOTION",
  "ball": {
    "x_px": 640.5,
    "y_px": 400.2,
    "radius_px": 12.0,
    "confidence": 0.95
  },
  "velocity": {
    "vx_px_s": 1500.0,
    "vy_px_s": -50.0,
    "speed_px_s": 1500.8
  },
  "metrics": {
    "cap_fps": 119.8,
    "proc_fps": 118.5,
    "disp_fps": 60.0,
    "proc_latency_ms": 2.1,
    "idle_stddev": 0.8
  },
  "shot": {
    "speed_m_s": 2.45,
    "direction_deg": 1.2,
    "frames_to_speed": 4
  },
  "calibrated": true,
  "resolution": [1280, 800]
}
```

## License

Proprietary - StrikeLab
