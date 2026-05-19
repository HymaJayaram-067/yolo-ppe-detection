"""
Flask + WebSocket Server for Real-Time PPE Compliance Streaming
Streams annotated video frames to browser via SocketIO.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import base64
import threading
from tracker import PPEDetector

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global detector instance
detector = None
is_streaming = False


@app.route("/")
def index():
    return render_template("index.html")


def generate_frames(source):
    """Generate annotated frames and emit via WebSocket"""
    global detector, is_streaming

    detector = PPEDetector(model_path="yolov8n.pt", conf_threshold=0.4, mode="pretrained")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        socketio.emit("error", {"msg": f"Cannot open source: {source}"})
        return

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector.set_safety_zone(height)
    is_streaming = True

    while is_streaming:
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        tracks = detector.detect_and_track(frame)
        detector.check_zone_violations(tracks)
        annotated = detector.draw_annotations(frame, tracks)

        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        socketio.emit("frame", {
            "image": frame_b64,
            "stats": {
                "tracked": detector.total_tracked,
                "in_zone": len(detector.crossed_in),
                "violations": detector.violation_count,
            }
        })
        socketio.sleep(0.033)  # ~30 FPS

    cap.release()


@socketio.on("start_stream")
def handle_start(data):
    global is_streaming
    source = data.get("source", "input_video.mp4")
    if not is_streaming:
        threading.Thread(target=generate_frames, args=(source,), daemon=True).start()


@socketio.on("stop_stream")
def handle_stop():
    global is_streaming
    is_streaming = False


if __name__ == "__main__":
    print("Starting PPE Compliance Monitor Server...")
    print("Open http://localhost:5000 in your browser")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
