import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import VGG19_Weights
from PIL import Image
import gc
import torch.nn.functional as F
from scipy.ndimage import label, find_objects
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32,expandable_segments:True'

class MotionDetector:
    def __init__(self, threshold=45, min_area=256):   
        self.threshold = threshold
        self.min_area = min_area
    
    def detect_motion_regions(self, frame1, frame2):
        """
        Detect regions with significant motion between two frames.
        Returns list of (x, y, w, h) coordinates for motion regions.
        """
         
        diff = torch.abs(frame1.mean(dim=1, keepdim=True) - 
                        frame2.mean(dim=1, keepdim=True))
        
         
        motion_mask = (diff > self.threshold/255.0).float()
        
         
        from scipy.ndimage import binary_dilation, binary_erosion
        
        regions = []
        batch_size = motion_mask.size(0)
        
        for b in range(batch_size):
            mask = motion_mask[b, 0].cpu().numpy()
            
             
            mask = binary_dilation(mask, iterations=2)
            mask = binary_erosion(mask, iterations=1)
            
             
            labeled, num_features = label(mask)
            
            for region in find_objects(labeled):
                if region is not None:
                    y_start, y_end = region[0].start, region[0].stop
                    x_start, x_end = region[1].start, region[1].stop
                    area = (y_end - y_start) * (x_end - x_start)
                    
                    if area >= self.min_area:
                         
                        pad = 24   
                        y_start = max(0, y_start - pad)
                        y_end = min(mask.shape[0], y_end + pad)
                        x_start = max(0, x_start - pad)
                        x_end = min(mask.shape[1], x_end + pad)
                        
                        regions.append((b, x_start, y_start, x_end - x_start, y_end - y_start))
        
        return regions

def create_smooth_mask(height, width, device):
    """Create a smooth blending mask for the patches"""
    y = torch.linspace(0, 1, height, device=device)
    x = torch.linspace(0, 1, width, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    
     
    edge_falloff = 0.1
    mask = torch.ones_like(xx)
    mask *= torch.minimum(torch.sigmoid((xx - edge_falloff) / 0.1),
                         torch.sigmoid((1 - xx - edge_falloff) / 0.1))
    mask *= torch.minimum(torch.sigmoid((yy - edge_falloff) / 0.1),
                         torch.sigmoid((1 - yy - edge_falloff) / 0.1))
    
    return mask.unsqueeze(0).repeat(3, 1, 1)

class MotionAwareInterpolation(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_detector = MotionDetector()
        
         
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 32, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv2d(32, 3, 3, 1, 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, frame1, frame3):
        batch_size = frame1.size(0)
        device = frame1.device
        
        print("\nProcessing batch...")
        print(f"Input shape: {frame1.shape}")
        
         
        output = (frame1 + frame3) / 2
        print("Initialized output frame with average")
        
         
        motion_regions = self.motion_detector.detect_motion_regions(frame1, frame3)
        print(f"Detected {len(motion_regions)} motion regions")
        
         
        for idx, (batch_idx, x, y, w, h) in enumerate(motion_regions):
            print(f"\rProcessing region {idx+1}/{len(motion_regions)}: size {w}x{h} at ({x},{y})", end="")
            
             
            region1 = frame1[batch_idx:batch_idx+1, :, y:y+h, x:x+w]
            region3 = frame3[batch_idx:batch_idx+1, :, y:y+h, x:x+w]
            
             
            region_input = torch.cat((region1, region3), 1)
            
             
            features = self.feature_extractor(region_input)
            region_output = self.final_conv(features)
            
             
            mask = create_smooth_mask(h, w, device)
            output[batch_idx, :, y:y+h, x:x+w] = \
                output[batch_idx, :, y:y+h, x:x+w] * (1 - mask) + \
                region_output * mask
        
        print("\nCompleted region processing")
        return output.clamp(-1, 1)

class Vimeo90KDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        list_file = 'tri_trainlist.txt' if split == 'train' else 'tri_testlist.txt'
        with open(os.path.join(root_dir, list_file), 'r') as f:
            self.sequences = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_path = os.path.join(self.root_dir, 'sequences', self.sequences[idx])
        
        frame1 = Image.open(os.path.join(seq_path, 'im1.png'))
        frame2 = Image.open(os.path.join(seq_path, 'im2.png'))
        frame3 = Image.open(os.path.join(seq_path, 'im3.png'))
        
        if self.transform:
            frame1 = self.transform(frame1)
            frame2 = self.transform(frame2)
            frame3 = self.transform(frame3)
            
        return frame1, frame3, frame2

class CombinedLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval().to(device)
        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:2]),     
            nn.Sequential(*list(vgg.children())[2:7]),    
            nn.Sequential(*list(vgg.children())[7:12]),   
            nn.Sequential(*list(vgg.children())[12:21])   
        ]).eval()
        
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, pred, target):
         
        pred = pred.clamp(-1, 1)
        target = target.clamp(-1, 1)
        
         
        l1_loss = F.l1_loss(pred, target)
        
         
        pred_features = []
        target_features = []
        
        x_pred = pred
        x_target = target
        
        for slice in self.slices:
            x_pred = slice(x_pred)
            x_target = slice(x_target)
            pred_features.append(x_pred)
            target_features.append(x_target)
            
         
        perceptual_loss = 0
        for p, t in zip(pred_features, target_features):
            p = p.clamp(-1e3, 1e3)
            t = t.clamp(-1e3, 1e3)
            perceptual_loss += F.l1_loss(p, t)
        
         
        perceptual_loss = perceptual_loss * 0.01
        
         
        total_loss = l1_loss + perceptual_loss
        
        return total_loss, l1_loss, perceptual_loss

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU installation.")
    
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    BATCH_SIZE = 3
    
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    
    model = MotionAwareInterpolation().to(device)
    scaler = torch.cuda.amp.GradScaler()
    
    criterion = CombinedLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print("Loading datasets...")
    train_dataset = Vimeo90KDataset(
        root_dir='vimeo_triplet/vimeo_triplet',
        split='train',
        transform=transform
    )
    test_dataset = Vimeo90KDataset(
        root_dir='vimeo_triplet/vimeo_triplet',
        split='test',
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    print(f"Datasets loaded. Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    num_epochs = 2
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        valid_steps = 0
        
        for i, (frame1, frame3, target) in enumerate(train_loader):
            frame1, frame3, target = frame1.to(device), frame3.to(device), target.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(frame1, frame3)
                loss, l1, perceptual = criterion(outputs, target)
            
            if torch.isfinite(loss) and loss.item() > 0:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                total_train_loss += loss.item()
                valid_steps += 1
            
            if i % 10 == 0:   
                print(f'\nEpoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}]')
                print(f'Total Loss: {loss.item():.4f}, L1: {l1.item():.4f}, Perceptual: {perceptual.item():.4f}')
                print(f'GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB')
                print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
                
                torch.cuda.empty_cache()
                gc.collect()
        
        if valid_steps > 0:
            avg_train_loss = total_train_loss / valid_steps
            scheduler.step(avg_train_loss)
        
        model.eval()
        total_val_loss = 0
        valid_val_steps = 0
        
        print("\nStarting validation...")
        with torch.no_grad():
            for i, (frame1, frame3, target) in enumerate(test_loader):
                frame1, frame3, target = frame1.to(device), frame3.to(device), target.to(device)
                
                with torch.cuda.amp.autocast():
                    outputs = model(frame1, frame3)
                    loss, _, _ = criterion(outputs, target)
                
                if torch.isfinite(loss) and loss.item() > 0:
                    total_val_loss += loss.item()
                    valid_val_steps += 1
                
                if i % 100 == 0:
                    print(f'Validation Step [{i}/{len(test_loader)}], '
                          f'Loss: {loss.item():.4f}')
        
        if valid_val_steps > 0:
            avg_val_loss = total_val_loss / valid_val_steps
            
            print(f'\nEpoch [{epoch+1}/{num_epochs}]')
            print(f'Average Training Loss: {avg_train_loss:.4f}')
            print(f'Average Validation Loss: {avg_val_loss:.4f}')
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                print(f'New best model found! Validation Loss: {best_loss:.4f}')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                }, 'motion_aware_model.pth')

        torch.cuda.empty_cache()
        gc.collect()

    print("Training completed!")
    print(f"Best validation loss: {best_loss:.4f}")

if __name__ == '__main__':
    main()