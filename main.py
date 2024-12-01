import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.nn.functional import interpolate as F_interpolate
from PIL import Image
import gc
import torch.nn.functional as F
# Set CUDA device specifically to RTX 4070
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first (and probably only) GPU
torch.backends.cudnn.benchmark = True  # Enable CUDNN benchmarking for better performance

class FrameInterpolationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Rest of the model definition remains the same
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

def main():
    # Verify CUDA is available and print GPU info
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU installation.")
    
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Calculate optimal batch size based on GPU memory
    # RTX 4070 laptop has 8GB VRAM, so we'll be conservative
    BATCH_SIZE = 6  # Start with a conservative batch size
    
    # Set device
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()  # Clear GPU memory before starting
    
    # Initialize model with mixed precision training
    model = FrameInterpolationModel().to(device)
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Define transforms with GPU memory pinning
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load datasets
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

    # Create dataloaders with GPU memory pinning
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"Datasets loaded. Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Training loop
    num_epochs = 3
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for i, (frame1, frame3, target) in enumerate(train_loader):
            # Move data to GPU
            frame1, frame3, target = frame1.to(device, non_blocking=True), frame3.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            # Concatenate input frames
            inputs = torch.cat((frame1, frame3), 1)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, target)
            
            # Mixed precision backward pass
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, '
                      f'GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB')
                
                # Clear GPU cache periodically
                if torch.cuda.memory_allocated() > 6 * 1024**3:
                    torch.cuda.empty_cache()
                    gc.collect()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        print("\nStarting validation...")
        with torch.no_grad():  # Disable gradient calculation for validation
            for i, (frame1, frame3, target) in enumerate(test_loader):
                # Move data to GPU
                frame1, frame3, target = frame1.to(device, non_blocking=True), frame3.to(device, non_blocking=True), target.to(device, non_blocking=True)
                
                # Concatenate input frames
                inputs = torch.cat((frame1, frame3), 1)
                
                # Forward pass
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, target)
                
                total_val_loss += loss.item()
                
                if i % 100 == 0:
                    print(f'Validation Step [{i}/{len(test_loader)}], '
                          f'Loss: {loss.item():.4f}')
        
        avg_val_loss = total_val_loss / len(test_loader)
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        
        # Save best model based on validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            print(f'New best model found! Validation Loss: {best_loss:.4f}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, 'best_model.pth')

        # Clear GPU cache between epochs
        torch.cuda.empty_cache()
        gc.collect()

    print("Training completed!")
    print(f"Best validation loss: {best_loss:.4f}")

if __name__ == '__main__':
    main()