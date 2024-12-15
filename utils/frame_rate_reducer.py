import cv2

def reduce_fps(input_path, output_path, target_fps):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open input video")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(original_fps / target_fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_skip == 0:
            out.write(frame)
            
        frame_count += 1

    cap.release()
    out.release()

input_video = "input.avi"
output_video = "output.avi"
target_fps = 12

reduce_fps(input_video, output_video, target_fps)