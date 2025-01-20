import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from scipy.ndimage import binary_dilation, binary_erosion
import numpy as np

def load_image(path, transform):
    """Load and preprocess an image."""
    img = Image.open(path)
    return transform(img)

class MotionVisualizer:
    def __init__(self, threshold=45, min_area=256):
        self.threshold = threshold
        self.min_area = min_area
    
    def compute_motion_mask(self, frame1, frame2):
        """Compute and return all stages of motion mask computation."""
         
        diff = torch.abs(frame1.mean(dim=1, keepdim=True) - 
                        frame2.mean(dim=1, keepdim=True))
        
         
        initial_mask = (diff > self.threshold/255.0).float()
        
         
        processed_mask = initial_mask[0, 0].cpu().numpy()
        
         
        dilated_mask = binary_dilation(processed_mask, iterations=2)
        final_mask = binary_erosion(dilated_mask, iterations=1)
        
        return {
            'difference': diff[0, 0].cpu().numpy(),
            'initial_mask': processed_mask,
            'dilated_mask': dilated_mask,
            'final_mask': final_mask
        }

def visualize_motion_detection(frame1, frame3, save_path):
    """Create visualization of motion detection process."""
    motion_viz = MotionVisualizer()
    masks = motion_viz.compute_motion_mask(frame1, frame3)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.3)
    
     
    axes[0, 0].imshow(tensor_to_image(frame1))
    axes[0, 0].set_title('Frame 1')
    axes[0, 1].imshow(tensor_to_image(frame3))
    axes[0, 1].set_title('Frame 3')
    
     
    diff_map = masks['difference']
    diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min())
    axes[0, 2].imshow(diff_map, cmap='hot')
    axes[0, 2].set_title('Difference Map')
    
     
    axes[1, 0].imshow(masks['initial_mask'], cmap='gray')
    axes[1, 0].set_title(f'Initial Mask\n(threshold={motion_viz.threshold}/255)')
    
     
    axes[1, 1].imshow(masks['dilated_mask'], cmap='gray')
    axes[1, 1].set_title('After Dilation')
    
     
    axes[1, 2].imshow(masks['final_mask'], cmap='gray')
    axes[1, 2].set_title('Final Mask\n(After Erosion)')
    
     
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def tensor_to_image(tensor):
    """Convert a normalized tensor to a displayable image."""
    tensor = tensor.clone().detach()
    tensor = tensor * 0.5 + 0.5
    tensor = tensor.clamp(0, 1)
    
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    return tensor.permute(1, 2, 0).numpy()

def main():
     
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
     
    test_folders = ['test/0110', 'test/0728']
    
    for folder in test_folders:
        print(f"\nProcessing sequence in {folder}")
        
         
        frame1 = load_image(os.path.join(folder, 'im1.png'), transform)
        frame3 = load_image(os.path.join(folder, 'im3.png'), transform)
        
         
        frame1 = frame1.unsqueeze(0)
        frame3 = frame3.unsqueeze(0)
        
         
        save_path = os.path.join(folder, 'motion_mask.png')
        visualize_motion_detection(frame1, frame3, save_path)
        print(f"Motion mask visualization saved to {save_path}")

if __name__ == '__main__':
    main()