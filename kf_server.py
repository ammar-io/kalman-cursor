#!/usr/bin/env python3
"""
kf_server.py
Serve Kalman-filtered cursor positions over HTTP/SSE with improved tracking.
"""
import cv2
import numpy as np
import pyautogui          # captures raw cursor pos
from flask import Flask, Response, send_file
import json, time, threading, pathlib


class KalmanFilterTracker:
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        # Create Kalman filter object - 4 dynamic parameters (x, y, dx, dy), 2 measurement parameters (x, y)
        self.kalman = cv2.KalmanFilter(4, 2)
        
        # Set transition matrix (state transition model)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x = x + dx
            [0, 1, 0, 1],  # y = y + dy
            [0, 0, 1, 0],  # dx = dx
            [0, 0, 0, 1]   # dy = dy
        ], dtype=np.float32)
        
        # Set measurement matrix
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],  # We measure x
            [0, 1, 0, 0]   # We measure y
        ], dtype=np.float32)
        
        # Set process noise covariance
        self.kalman.processNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32) * process_noise
        
        # Set measurement noise covariance
        self.kalman.measurementNoiseCov = np.array([
            [1, 0],
            [0, 1]
        ], dtype=np.float32) * measurement_noise
        
        # Set error covariance matrix
        self.kalman.errorCovPost = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Initialize state
        self.kalman.statePost = np.array([0, 0, 0, 0], dtype=np.float32)
        
        # Track if we've had a measurement yet
        self.has_measurement = False
        
        # Last time we had a measurement
        self.last_measurement_time = 0
        
        # Max time to consider the object occluded
        self.max_occlusion_time = 2.0  # seconds
        
        # Is the object currently occluded
        self.is_occluded = False
        
        # Time step for the model
        self.dt = 1/60.0  # assuming 60Hz update rate
        
        # Adjust transition matrix for time step
        self.kalman.transitionMatrix[0, 2] = self.dt
        self.kalman.transitionMatrix[1, 3] = self.dt
    
    def reset(self):
        """Reset the Kalman filter"""
        self.kalman.statePost = np.array([0, 0, 0, 0], dtype=np.float32)
        self.has_measurement = False
    
    def predict(self):
        """Predict the next state"""
        prediction = self.kalman.predict()
        return prediction[:2].flatten()
    
    def update(self, measurement):
        """Update the state with a new measurement"""
        self.last_measurement_time = time.time()
        self.is_occluded = False
        
        if not self.has_measurement:
            # Initialize the state with the first measurement
            self.kalman.statePost = np.array([measurement[0], measurement[1], 0, 0], dtype=np.float32)
            self.has_measurement = True
            return measurement
        
        # Convert measurement to numpy array
        measurement_array = np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]])
        
        # Correct the state using the measurement
        self.kalman.correct(measurement_array)
        
        # Return the corrected state
        return self.kalman.statePost[:2].flatten()
    
    def check_occlusion(self):
        """Check if the tracked object is occluded"""
        if time.time() - self.last_measurement_time > self.max_occlusion_time:
            self.is_occluded = True
        return self.is_occluded


# Initialize the Kalman filter tracker
tracker = KalmanFilterTracker(process_noise=1e-3, measurement_noise=1e-1)

state_lock = threading.Lock()       # protect concurrent access
latest_pred = np.zeros((2,), np.float32)
latest_raw = np.zeros((2,), np.float32)
is_occluded = False

# ─────────────────────────────────────────────────────────────
def tracker_loop():
    """Background thread: read OS cursor → correct → predict"""
    global latest_pred, latest_raw, is_occluded
    while True:
        x, y = pyautogui.position()                # raw measurement (pixels)
        latest_raw[:] = (x, y)                     # store raw for streaming
        
        with state_lock:
            # Update the Kalman filter with the new measurement
            tracker.update((x, y))
            
            # Get prediction for next frame
            latest_pred[:] = tracker.predict()
            
            # Check if the object is occluded
            is_occluded = tracker.check_occlusion()
            
        time.sleep(tracker.dt)

# 2.  Minimal Flask API
app = Flask(__name__)
BASE = pathlib.Path(__file__).parent

@app.route("/pos")
def pos_json():
    """Return the latest filtered coordinates as JSON."""
    with state_lock:
        x, y = map(float, latest_pred)
        occluded = is_occluded
    return {"x": x, "y": y, "occluded": occluded}

@app.route("/stream")
def sse():
    """Server-Sent Events stream → convenient for JS EventSource."""
    def gen():
        while True:
            with state_lock:
                payload = json.dumps({
                    "raw": [float(latest_raw[0]), float(latest_raw[1])],
                    "kal": [float(latest_pred[0]), float(latest_pred[1])],
                    "occluded": is_occluded
                })
            yield f"data:{payload}\n\n"
            time.sleep(tracker.dt)
    return Response(gen(), mimetype="text/event-stream")

@app.route("/overlay")
def overlay_html():
    """Serve the static overlay page that OBS will load."""
    return send_file(BASE / "overlay.html")

from flask import redirect
@app.route("/")
def index():
    return redirect("/overlay", code=302)

@app.route("/config", methods=["GET", "POST"])
def config():
    """Allow configuration of the Kalman filter parameters."""
    from flask import request, jsonify
    
    if request.method == "POST":
        data = request.json
        with state_lock:
            if "process_noise" in data:
                tracker.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * float(data["process_noise"])
            if "measurement_noise" in data:
                tracker.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * float(data["measurement_noise"])
            if "max_occlusion_time" in data:
                tracker.max_occlusion_time = float(data["max_occlusion_time"])
        return jsonify({"status": "updated"})
    else:
        with state_lock:
            return jsonify({
                "process_noise": float(tracker.kalman.processNoiseCov[0, 0]),
                "measurement_noise": float(tracker.kalman.measurementNoiseCov[0, 0]),
                "max_occlusion_time": tracker.max_occlusion_time
            })


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=tracker_loop, daemon=True).start()
    app.run("0.0.0.0", port=5001, threaded=True)