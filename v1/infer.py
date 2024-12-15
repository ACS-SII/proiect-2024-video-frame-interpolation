import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import time

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

def load_model(model_path):
    model = FrameInterpolationModel()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def interpolate_frame(model, frame1_path, frame3_path, output_path, device='cuda'):
    total_start = time.time()
    

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    preprocess_start = time.time()

    frame1 = Image.open(frame1_path)
    frame3 = Image.open(frame3_path)
    
    frame1 = transform(frame1).unsqueeze(0)
    frame3 = transform(frame3).unsqueeze(0)

    frame1 = frame1.to(device)
    frame3 = frame3.to(device)
    model = model.to(device)
    

    inputs = torch.cat((frame1, frame3), 1)
    preprocess_time = time.time() - preprocess_start

    with torch.no_grad():
        torch.cuda.synchronize()
        inference_start = time.time()
        output = model(inputs)
        torch.cuda.synchronize()
        inference_time = time.time() - inference_start
    
    postprocess_start = time.time()
    output = output * 0.5 + 0.5
    output = torch.clamp(output, 0, 1)
    output_img = transforms.ToPILImage()(output.squeeze(0).cpu())
    
    output_img.save(output_path)
    postprocess_time = time.time() - postprocess_start
    
    total_time = time.time() - total_start
    
    print(f"\nTiming Information:")
    print(f"Preprocessing time: {preprocess_time*1000:.2f}ms")
    print(f"Model inference time: {inference_time*1000:.2f}ms")
    print(f"Postprocessing time: {postprocess_time*1000:.2f}ms")
    print(f"Total processing time: {total_time*1000:.2f}ms")
    print(f"Frames per second (FPS): {1/inference_time:.2f}")
    print(f"\nInterpolated frame saved to: {output_path}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model_path = 'best_model.pth'
    model = load_model(model_path)
    
    frame1_path = '0001/im1.png'
    frame3_path = '0001/im3.png'
    output_path = 'interpolated_frame.png'
    
    print("\nWarm-up run:")
    interpolate_frame(model, frame1_path, frame3_path, output_path, device)
    
    print("\nTiming run:")
    interpolate_frame(model, frame1_path, frame3_path, output_path, device)

if __name__ == '__main__':
    main()