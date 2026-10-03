"""Generates a synthetic exam hall video clip for offline pipeline verification."""

import os
import cv2
import numpy as np


def generate_sample_exam_video(
    output_path: str = "tests/fixtures/sample_exam.mp4",
    duration_seconds: float = 6.0,
    fps: float = 15.0,
    width: int = 640,
    height: int = 480,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(duration_seconds * fps)

    for i in range(total_frames):
        # Create exam room background (classroom floor & desk)
        frame = np.full((height, width, 3), (220, 220, 225), dtype=np.uint8)

        # Draw desk
        cv2.rectangle(frame, (120, 280), (520, 440), (160, 140, 120), -1)
        cv2.rectangle(frame, (120, 280), (520, 440), (100, 80, 60), 2)

        # Draw exam paper on desk
        cv2.rectangle(frame, (260, 310), (380, 410), (250, 250, 250), -1)
        cv2.rectangle(frame, (260, 310), (380, 410), (180, 180, 180), 1)

        # Draw student (body)
        cv2.ellipse(frame, (320, 260), (90, 110), 0, 0, 360, (70, 70, 140), -1)

        # Draw head with slight motion
        head_x = 320
        head_y = 150
        if i > int(fps * 2.0):  # Student turns head after 2 seconds
            head_x = 335

        cv2.circle(frame, (head_x, head_y), 45, (190, 205, 235), -1)
        cv2.circle(frame, (head_x, head_y), 45, (120, 130, 160), 2)

        # Draw phone on desk after 1.5 seconds
        if i >= int(fps * 1.5):
            cv2.rectangle(frame, (390, 330), (430, 390), (30, 30, 30), -1)
            cv2.rectangle(frame, (392, 332), (428, 388), (200, 220, 255), -1)

        writer.write(frame)

    writer.release()
    print(f"Generated sample video '{output_path}' ({total_frames} frames @ {fps} FPS)")


if __name__ == "__main__":
    generate_sample_exam_video()
