import cv2
import argparse
import os
from datetime import timedelta

def extract_frames(video_path, timestamp_seconds, output_dir="extracted_frames", num_frames=3, frame_offset=0):
    """
    Extract frames from a video starting at a specific timestamp.
    
    Args:
        video_path (str): Path to the input video file
        timestamp_seconds (float): Time in seconds to start extraction
        output_dir (str): Directory to save extracted frames
        num_frames (int): Number of frames to extract (default: 3)
        frame_offset (int): Number of frames to skip before starting extraction (default: 0)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    target_frame = int(timestamp_seconds * fps)
    
    print(f"Video FPS: {fps}")
    print(f"Video duration: {timedelta(seconds=int(duration))}")
    print(f"Starting extraction at: {timedelta(seconds=int(timestamp_seconds))}")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame + frame_offset)
    
    frames_saved = 0
    while frames_saved < num_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"Could only extract {frames_saved} frames before reaching end of video")
            break
            
        frame_number = target_frame + frame_offset + frames_saved
        output_path = os.path.join(output_dir, f"frame_{frame_number:06d}.png")
        cv2.imwrite(output_path, frame)
        print(f"Saved frame {frames_saved + 1}/{num_frames} to {output_path}")
        frames_saved += 1
    
    cap.release()
    print("Frame extraction completed!")

def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video at a specific timestamp")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("timestamp", type=float, help="Time in seconds to start extraction")
    parser.add_argument("--output-dir", default="extracted_frames", help="Directory to save extracted frames")
    parser.add_argument("--num-frames", type=int, default=3, help="Number of frames to extract")
    parser.add_argument("--frame-offset", type=int, default=0, help="Number of frames to skip before starting extraction")
    
    args = parser.parse_args()
    
    try:
        extract_frames(
            args.video_path,
            args.timestamp,
            args.output_dir,
            args.num_frames,
            args.frame_offset
        )
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()