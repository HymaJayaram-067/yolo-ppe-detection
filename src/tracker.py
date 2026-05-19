"""
PPE Compliance Detection with YOLOv8 + DeepSORT Tracking
Detects PPE violations (missing hardhat/vest) and tracks workers across frames.
Supports both pretrained YOLO (person detection) and fine-tuned PPE model.
"""
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


class PPEDetector:
    """Real-time PPE compliance detector with multi-object tracking."""

    # PPE class mapping (for fine-tuned model)
    PPE_CLASSES = {0: "Hardhat", 1: "No-Hardhat", 2: "Safety-Vest", 3: "No-Safety-Vest"}
    # COCO fallback (pretrained model)
    COCO_PERSON_CLASS = 0

    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5, mode="pretrained"):
        """
        Args:
            model_path: Path to YOLO weights (.pt file)
            conf_threshold: Minimum confidence for detections
            mode: 'pretrained' (COCO person detection) or 'ppe' (fine-tuned PPE model)
        """
        self.model = YOLO(model_path)
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_iou_distance=0.7,
        )
        self.conf_threshold = conf_threshold
        self.mode = mode
        # Violation tracking
        self.violations = {}  # track_id -> violation type
        self.violation_count = 0
        self.total_tracked = 0
        # Zone-based counting
        self.crossed_in = set()
        self.crossed_out = set()

    def set_safety_zone(self, frame_height, zone_start=0.3, zone_end=0.8):
        """Define the safety zone where PPE is required"""
        self.zone_top = int(frame_height * zone_start)
        self.zone_bottom = int(frame_height * zone_end)

    def detect_and_track(self, frame):
        """Run YOLO detection and DeepSORT tracking on a single frame"""
        results = self.model(frame, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue

            if self.mode == "ppe":
                # Fine-tuned model: detect all PPE classes
                if cls in self.PPE_CLASSES:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))
            else:
                # Pretrained model: detect persons only
                if cls == self.COCO_PERSON_CLASS:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        # Update tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)
        return tracks

    def check_zone_violations(self, tracks):
        """Check if tracked workers are inside safety zone without PPE"""
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_ltrb()
            center_y = (bbox[1] + bbox[3]) / 2

            # Check if person is in safety zone
            if self.zone_top <= center_y <= self.zone_bottom:
                if track_id not in self.crossed_in:
                    self.crossed_in.add(track_id)
                    self.total_tracked += 1
                # In PPE mode, check for violation classes
                if self.mode == "ppe" and hasattr(track, 'det_class'):
                    if track.det_class in [1, 3]:  # No-Hardhat or No-Safety-Vest
                        if track_id not in self.violations:
                            self.violations[track_id] = self.PPE_CLASSES[track.det_class]
                            self.violation_count += 1
                else:
                    # Pretrained mode: count zone entries/exits
                    if hasattr(track, 'prev_y'):
                        if track.prev_y < self.zone_top and center_y >= self.zone_top:
                            self.crossed_in.add(track_id)
                        elif track.prev_y > self.zone_bottom and center_y <= self.zone_bottom:
                            self.crossed_out.add(track_id)
            track.prev_y = center_y

    def draw_annotations(self, frame, tracks):
        """Draw bounding boxes, track IDs, safety zone, and violation alerts"""
        h, w = frame.shape[:2]

        # Draw safety zone
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, self.zone_top), (w, self.zone_bottom), (0, 100, 255), -1)
        frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
        cv2.line(frame, (0, self.zone_top), (w, self.zone_top), (0, 255, 255), 2)
        cv2.line(frame, (0, self.zone_bottom), (w, self.zone_bottom), (0, 255, 255), 2)
        cv2.putText(frame, "SAFETY ZONE", (10, self.zone_top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw tracks
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = [int(v) for v in bbox]

            # Color based on violation status
            if track_id in self.violations:
                color = (0, 0, 255)  # Red for violation
                label = f"ID:{track_id} VIOLATION"
            elif track_id in self.crossed_in:
                color = (0, 255, 0)  # Green for compliant
                label = f"ID:{track_id}"
            else:
                color = (255, 150, 0)  # Blue for outside zone
                label = f"ID:{track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw stats panel
        cv2.rectangle(frame, (5, 5), (280, 110), (0, 0, 0), -1)
        cv2.putText(frame, f"Workers Tracked: {self.total_tracked}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"In Zone: {len(self.crossed_in)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Violations: {self.violation_count}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Mode: {self.mode.upper()}", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def process_video(self, source, output_path="output/result.mp4", show=False):
        """Process entire video file for PPE compliance detection"""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Cannot open video source: {source}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.set_safety_zone(height)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        print(f"[PPE Detector] Processing: {source}")
        print(f"  Resolution: {width}x{height} | FPS: {fps} | Frames: {total_frames}")
        print(f"  Mode: {self.mode} | Confidence: {self.conf_threshold}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  Frame {frame_count}/{total_frames} | "
                      f"Tracked: {self.total_tracked} | Violations: {self.violation_count}")

            tracks = self.detect_and_track(frame)
            self.check_zone_violations(tracks)
            annotated = self.draw_annotations(frame, tracks)
            out.write(annotated)

            if show:
                cv2.imshow("PPE Compliance Monitor", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        out.release()
        if show:
            cv2.destroyAllWindows()

        print(f"\n{'='*50}")
        print(f"  RESULTS")
        print(f"{'='*50}")
        print(f"  Total Workers Tracked: {self.total_tracked}")
        print(f"  Zone Entries: {len(self.crossed_in)}")
        print(f"  Violations Detected: {self.violation_count}")
        print(f"  Output: {output_path}")
        print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPE Compliance Detection System")
    parser.add_argument("--source", required=True, help="Path to video file or camera index (0)")
    parser.add_argument("--output", default="output/result.mp4", help="Output video path")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--mode", choices=["pretrained", "ppe"], default="pretrained",
                        help="'pretrained' for COCO person detection, 'ppe' for fine-tuned model")
    parser.add_argument("--show", action="store_true", help="Display video while processing")
    args = parser.parse_args()

    detector = PPEDetector(model_path=args.model, conf_threshold=args.conf, mode=args.mode)
    detector.process_video(args.source, args.output, show=args.show)
