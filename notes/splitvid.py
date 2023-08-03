import os
import cv2

def split_video_into_frames(input_video_path, output_frames_dir):
    # Read input video
    video = cv2.VideoCapture(input_video_path)

    # Create output directory if it doesn't exist
    os.makedirs(output_frames_dir, exist_ok=True)

    # Variables for frame indexing
    frame_count = 0
    success = True

    # Split video into frames
    while success:
        # Read frame from video
        success, frame = video.read()

        if success:
            # Save frame as image file
            frame_path = os.path.join(output_frames_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)

            frame_count += 1

    # Release resources
    video.release()

if __name__ == "__main__":
    input_video_path = r"C:\Users\mikus\lab\input\cam1.mp4"  # Specify input video file path
    output_frames_dir = r"C:\Users\mikus\lab\output"  # Specify output frames directory path
    split_video_into_frames(input_video_path, output_frames_dir)
