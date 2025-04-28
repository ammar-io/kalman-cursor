#!/usr/bin/env python3
"""
kf_server.py
Serve Kalman-filtered cursor positions over HTTP/SSE.
"""
import cv2
import numpy as np
import pyautogui          # captures raw cursor pos
from flask import Flask, Response, send_file
import json, time, threading, pathlib


# 1.  Kalman filter: constant-velocity (x, y, vx, vy) model

DT = 1/60.0                          # model time-step (sec) – assume 60 Hz
kf = cv2.KalmanFilter(4, 2)          # 4-state, 2-measurement

kf.transitionMatrix = np.array([[1, 0, DT, 0],
                                [0, 1, 0, DT],
                                [0, 0, 1,  0],
                                [0, 0, 0,  1]], np.float32)

kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], np.float32)

kf.processNoiseCov    = np.eye(4, dtype=np.float32) * 1e-3  # tune ↓ to smooth more
kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1 # tune ↓ to trust sensor more
kf.statePost          = np.zeros((4, 1), np.float32)        # initial state

state_lock = threading.Lock()       # protect concurrent access
latest_pred = np.zeros((2,), np.float32)
latest_raw   = np.zeros((2,), np.float32)



# ─────────────────────────────────────────────────────────────
def tracker_loop():
    """Background thread: read OS cursor → correct → predict"""
    global latest_pred
    while True:
        x, y = pyautogui.position()                # raw measurement (pixels)
        latest_raw[:] = (x, y)                     # store raw for streaming
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        with state_lock:
            kf.correct(meas)                       # incorporate measurement
            pred = kf.predict()                    # best estimate for *next* frame
            latest_pred[:] = pred[:2, 0]           # store (x,y) only
        time.sleep(DT)

# 2.  Minimal Flask API
app = Flask(__name__)
BASE = pathlib.Path(__file__).parent

@app.route("/pos")
def pos_json():
    """Return the latest filtered coordinates as JSON."""
    with state_lock:
        x, y = map(float, latest_pred)
    return {"x": x, "y": y}

@app.route("/stream")
def sse():
    """Server-Sent Events stream → convenient for JS EventSource."""
    def gen():
        while True:
            with state_lock:
                payload = json.dumps({
                    "raw" : [ float(latest_raw[0]),  float(latest_raw[1]) ],
                    "kal" : [ float(latest_pred[0]), float(latest_pred[1]) ]
                })
            yield f"data:{payload}\n\n"
            time.sleep(DT)
    return Response(gen(), mimetype="text/event-stream")

@app.route("/overlay")
def overlay_html():
    """Serve the static overlay page that OBS will load."""
    return send_file(BASE / "overlay.html")



# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=tracker_loop, daemon=True).start()
    app.run("0.0.0.0", port=5001, threaded=True)
