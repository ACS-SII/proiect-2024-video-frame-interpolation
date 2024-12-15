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

def interpolate_frame_tensor(model, frame1_tensor, frame2_tensor, device):
    with torch.no_grad():
        inputs = torch.cat((frame1_tensor, frame2_tensor), 1)
        
        output = model(inputs)
        
        output = output * 0.5 + 0.5
        output = torch.clamp(output, 0, 1)
        
        return output

def process_video(input_path, output_path, model, device, target_size=(448, 256)):
    print("\nOpening input video...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video FPS: {fps}")
    print(f"Frame count: {frame_count}")
    print(f"Target output FPS: {fps*2}")
    
    print("\nCreating output video writer...")
    fourcc = cv2.VideoWriter_fourcc(*'RGBA')
    out = cv2.VideoWriter(output_path, fourcc, fps*2, target_size)
    
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Could not create output video writer: {output_path}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    try:
        print("\nReading first frame...")
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")
        
        
        pbar = tqdm(total=frame_count-1, desc="Processing frames")
        
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
                
            prev_frame_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
            curr_frame_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
            
            prev_tensor = transform(Image.fromarray(prev_frame_rgb)).unsqueeze(0).to(device)
            curr_tensor = transform(Image.fromarray(curr_frame_rgb)).unsqueeze(0).to(device)
            
            interpolated = interpolate_frame_tensor(model, prev_tensor, curr_tensor, device)
            
            interpolated_np = (interpolated.squeeze(0).cpu().numpy() * 255).transpose(1, 2, 0).astype(np.uint8)
            interpolated_bgr = cv2.cvtColor(interpolated_np, cv2.COLOR_RGB2BGR)
            
            out.write(prev_frame)
            out.write(interpolated_bgr)
            prev_frame = curr_frame
            
            pbar.update(1)
        
        out.write(prev_frame)
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        pbar.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()

def main():
    print("Initializing...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    model_path = 'best_model.pth'
    print(f"\nLoading model from: {model_path}")
    model = load_model(model_path, device)
    
    input_video = "input_12fps.avi"
    output_video = "output_interpolated.avi"
    
    print(f"\nInput video: {input_video}")
    print(f"Output video will be saved to: {output_video}")
    
    try:
        process_video(input_video, output_video, model, device)
        print(f"\nSuccess! Output saved to: {output_video}")
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        print("\nProcessing completed.")

if __name__ == "__main__":
    main()