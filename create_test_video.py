"""
Generate a simple test video with moving rectangles (simulating people walking)
so we can test the tracker without downloading anything.
"""
import cv2
import numpy as np
import os


def create_test_video(output_path="test_video.mp4", num_frames=150, width=640, height=480):
    """Create a synthetic video with moving objects to test tracking"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    # Simulate 3 "people" moving downward
    objects = [
        {"x": 100, "y": -80, "w": 40, "h": 80, "speed_y": 4, "speed_x": 1, "color": (0, 150, 255)},
        {"x": 300, "y": -200, "w": 45, "h": 90, "speed_y": 3, "speed_x": -1, "color": (255, 100, 0)},
        {"x": 500, "y": -50, "w": 35, "h": 75, "speed_y": 5, "speed_x": 0, "color": (0, 255, 100)},
    ]

    for frame_idx in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Background with some texture
        frame[:] = (40, 40, 40)

        # Draw counting line
        line_y = int(height * 0.6)
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 1)

        # Move and draw objects
        for obj in objects:
            obj["x"] += obj["speed_x"]
            obj["y"] += obj["speed_y"]

            # Reset if gone off screen
            if obj["y"] > height + 100:
                obj["y"] = -100

            x, y, w, h = int(obj["x"]), int(obj["y"]), obj["w"], obj["h"]
            if 0 <= y < height:
                cv2.rectangle(frame, (x, y), (x + w, y + h), obj["color"], -1)
                # Draw "head" circle
                cv2.circle(frame, (x + w // 2, y - 10), 15, obj["color"], -1)

        out.write(frame)

    out.release()
    print(f"Created test video: {output_path} ({num_frames} frames, {width}x{height})")


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    create_test_video()
