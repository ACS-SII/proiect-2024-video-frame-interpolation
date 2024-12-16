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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32,expandable_segments:True'

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
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feature_maps = [extractor(x) for extractor in self.feature_extractor]
        feature_maps_resized = [self.resize(fm) for fm in feature_maps]
        x = torch.cat(feature_maps_resized, 1)
        x = F.relu(self.fusion_conv(x))
        x = self.final_conv(x)
        return x.clamp(-1, 1)

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
            nn.Sequential(*list(vgg.children())[:2]),    # relu1_1
            nn.Sequential(*list(vgg.children())[2:7]),   # relu2_1
            nn.Sequential(*list(vgg.children())[7:12]),  # relu3_1
            nn.Sequential(*list(vgg.children())[12:21])  # relu4_1
        ]).eval()
        
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, pred, target):
        # Clamp inputs
        pred = pred.clamp(-1, 1)
        target = target.clamp(-1, 1)
        
        # L1 Loss
        l1_loss = F.l1_loss(pred, target)
        
        # VGG Perceptual Loss
        pred_features = []
        target_features = []
        
        x_pred = pred
        x_target = target
        
        for slice in self.slices:
            x_pred = slice(x_pred)
            x_target = slice(x_target)
            pred_features.append(x_pred)
            target_features.append(x_target)
            
        # Calculate perceptual loss with clamping
        perceptual_loss = 0
        for p, t in zip(pred_features, target_features):
            p = p.clamp(-1e3, 1e3)
            t = t.clamp(-1e3, 1e3)
            perceptual_loss += F.l1_loss(p, t)
        
        # Scale perceptual loss
        perceptual_loss = perceptual_loss * 0.01
        
        # Combine losses
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
    
    model = FrameInterpolationModel().to(device)
    scaler = torch.amp.GradScaler('cuda')
    
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

    num_epochs = 12
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        valid_steps = 0
        
        for i, (frame1, frame3, target) in enumerate(train_loader):
            frame1, frame3, target = frame1.to(device), frame3.to(device), target.to(device)
            
            inputs = torch.cat((frame1, frame3), 1)
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
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
            
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], '
                      f'Total Loss: {loss.item():.4f}, L1: {l1.item():.4f}, '
                      f'Perceptual: {perceptual.item():.4f}, '
                      f'GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB')
                
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
                
                inputs = torch.cat((frame1, frame3), 1)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
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
                }, 'combined_loss_model.pth')

        torch.cuda.empty_cache()
        gc.collect()

    print("Training completed!")
    print(f"Best validation loss: {best_loss:.4f}")

if __name__ == '__main__':
    main()