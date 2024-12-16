import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import time
from tqdm import tqdm

class FrameInterpolationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(6, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.ReLU(inplace=True)
            ) for _ in range(3)
        ])
        self.resize = nn.Upsample(size=(256, 448), mode='bilinear', align_corners=False)
        self.fusion_conv = nn.Conv2d(384, 128, 3, 1, 1)
        self.final_conv = nn.Conv2d(128, 3, 3, 1, 1)

    def forward(self, x):
        feature_maps = [extractor(x) for extractor in self.feature_extractor]
        feature_maps_resized = [self.resize(fm) for fm in feature_maps]
        x = torch.cat(feature_maps_resized, 1)
        x = F.relu(self.fusion_conv(x))
        x = self.final_conv(x)
        return x

def load_model(model_path, device):
    model = FrameInterpolationModel().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def split_frame_into_tiles(frame, tile_height=256, tile_width=448):
    height, width = frame.shape[:2]
    num_tiles_h = (height + tile_height - 1) // tile_height
    num_tiles_w = (width + tile_width - 1) // tile_width
    padded_height = num_tiles_h * tile_height
    padded_width = num_tiles_w * tile_width
    padded_frame = np.zeros((padded_height, padded_width, frame.shape[2]), dtype=frame.dtype)
    padded_frame[:height, :width] = frame
    tiles = []
    positions = []
    for i in range(num_tiles_h):
        for j in range(num_tiles_w):
            y = i * tile_height
            x = j * tile_width
            tile = padded_frame[y:y+tile_height, x:x+tile_width].copy()
            assert tile.shape == (tile_height, tile_width, 3), f"Incorrect tile shape: {tile.shape}"
            tiles.append(tile)
            positions.append((y, x))
    return tiles, positions, (padded_height, padded_width)

def reconstruct_frame_from_tiles(tiles, positions, original_size):
    height, width = original_size
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for tile, (y, x) in zip(tiles, positions):
        tile_height, tile_width = tile.shape[:2]
        write_height = min(tile_height, height - y)
        write_width = min(tile_width, width - x)
        frame[y:y+write_height, x:x+write_width] = tile[:write_height, :write_width]
    return frame

def interpolate_frame_tensor(model, frame1_tensor, frame2_tensor, device):
    with torch.no_grad():
        inputs = torch.cat((frame1_tensor, frame2_tensor), 1)
        output = model(inputs)
        output = output * 0.5 + 0.5
        output = torch.clamp(output, 0, 1)
        return output

def process_video(input_path, output_path, model, device, time_limit_seconds=17):
    print("\nOpening input video...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_limit = int(time_limit_seconds * fps)
    frame_count = min(total_frame_count, frame_limit)
    
    print(f"Input video size: {width}x{height}")
    print(f"Input video FPS: {fps}")
    print(f"Total frame count: {total_frame_count}")
    print(f"Processing first {time_limit_seconds} seconds ({frame_count} frames)")
    print(f"Target output FPS: {fps*3}")  # Now tripling the frame rate
    
    print("\nCreating output video writer...")
    temp_output = output_path.rsplit('.', 1)[0] + '_temp.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps*3, (width, height))  # Tripled FPS
    
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Could not create output video writer: {temp_output}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    frame_counter = 0
    try:
        print("\nReading first frame...")
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")
        
        pbar = tqdm(total=frame_count-1, desc="Processing frames")
        
        while frame_counter < frame_count - 1:
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            # Process the first pair of frames (prev_frame and curr_frame)
            prev_tiles, positions, padded_size = split_frame_into_tiles(
                cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
            )
            
            curr_tiles, _, _ = split_frame_into_tiles(
                cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
            )
            
            # Generate first interpolated frame
            interpolated_tiles_1 = []
            for prev_tile, curr_tile in zip(prev_tiles, curr_tiles):
                prev_tensor = transform(Image.fromarray(prev_tile)).unsqueeze(0).to(device)
                curr_tensor = transform(Image.fromarray(curr_tile)).unsqueeze(0).to(device)
                interpolated = interpolate_frame_tensor(model, prev_tensor, curr_tensor, device)
                interpolated_np = (interpolated.squeeze(0).cpu().numpy() * 255).transpose(1, 2, 0).astype(np.uint8)
                interpolated_tiles_1.append(interpolated_np)
            
            interpolated_frame_1 = reconstruct_frame_from_tiles(interpolated_tiles_1, positions, (height, width))
            interpolated_bgr_1 = cv2.cvtColor(interpolated_frame_1, cv2.COLOR_RGB2BGR)
            
            # Generate second interpolated frame using the first interpolated frame and curr_frame
            interpolated_frame_1_tiles, positions, padded_size = split_frame_into_tiles(
                interpolated_frame_1  # Already in RGB
            )
            
            # Generate second interpolated frame
            interpolated_tiles_2 = []
            for int1_tile, curr_tile in zip(interpolated_frame_1_tiles, curr_tiles):
                int1_tensor = transform(Image.fromarray(int1_tile)).unsqueeze(0).to(device)
                curr_tensor = transform(Image.fromarray(curr_tile)).unsqueeze(0).to(device)
                interpolated = interpolate_frame_tensor(model, int1_tensor, curr_tensor, device)
                interpolated_np = (interpolated.squeeze(0).cpu().numpy() * 255).transpose(1, 2, 0).astype(np.uint8)
                interpolated_tiles_2.append(interpolated_np)
            
            interpolated_frame_2 = reconstruct_frame_from_tiles(interpolated_tiles_2, positions, (height, width))
            interpolated_bgr_2 = cv2.cvtColor(interpolated_frame_2, cv2.COLOR_RGB2BGR)
            
            # Write all frames in sequence
            out.write(prev_frame)
            out.write(interpolated_bgr_1)
            out.write(interpolated_bgr_2)
            
            prev_frame = curr_frame
            frame_counter += 1
            
            pbar.update(1)
        
        # Write the final frame
        out.write(prev_frame)
        frame_counter += 1
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise
    
    finally:
        print(f"\nTotal frames processed: {frame_counter}")
        print("\nCleaning up...")
        pbar.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        if os.path.exists(temp_output):
            try:
                print("\nConverting to final format...")
                os.system(f'ffmpeg -i {temp_output} -c:v libx264 -crf 23 -preset medium {output_path}')
            except Exception as e:
                print(f"Error during conversion: {str(e)}")

def main():
    print("Initializing...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    model_path = 'best_model_12epochs.pth'
    print(f"\nLoading model from: {model_path}")
    model = load_model(model_path, device)
    
    input_video = "input_6fps.avi"
    output_video = "output_interpolated_triple.mp4"
    
    print(f"\nInput video: {input_video}")
    print(f"Output video will be saved to: {output_video}")
    
    try:
        process_video(input_video, output_video, model, device, time_limit_seconds=130)
        print(f"\nSuccess! Output saved to: {output_video}")
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        print("\nProcessing completed.")

if __name__ == "__main__":
    main()