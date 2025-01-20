import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import os
import time

from v1.main import FrameInterpolationModel as ModelV1
from v2.main import FrameInterpolationModel as ModelV2
from v3.main import MotionAwareInterpolation as ModelV3

def load_image(path, transform):
    """Load and preprocess an image."""
    img = Image.open(path)
    return transform(img)

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

def calculate_metrics(pred, target):
    """Calculate PSNR and SSIM metrics."""
    psnr = PeakSignalNoiseRatio()
    ssim = StructuralSimilarityIndexMeasure()
    
     
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
    if len(target.shape) == 3:
        target = target.unsqueeze(0)
    
     
    pred = pred * 0.5 + 0.5
    target = target * 0.5 + 0.5
    
    return {
        'PSNR': psnr(pred, target).item(),
        'SSIM': ssim(pred, target).item()
    }

def test_sequence(model, frame1, frame3, frame2_gt, device):
    """Run interpolation and calculate metrics."""
    model.eval()
    
     
    with torch.no_grad():
        if isinstance(model, ModelV3):
            _ = model(frame1.to(device), frame3.to(device))
        else:
            inputs = torch.cat((frame1.to(device), frame3.to(device)), 1)
            _ = model(inputs)
    torch.cuda.synchronize()
    
     
    start_time = time.time()
    with torch.no_grad():
        if isinstance(model, ModelV3):
            frame2_pred = model(frame1.to(device), frame3.to(device))
        else:
            inputs = torch.cat((frame1.to(device), frame3.to(device)), 1)
            frame2_pred = model(inputs)
    torch.cuda.synchronize()
    inference_time = time.time() - start_time
    
    metrics = calculate_metrics(frame2_pred.cpu(), frame2_gt)
    metrics['time'] = inference_time
    return frame2_pred.cpu(), metrics

def visualize_results(frame1, frame2_gt, frame3, predictions, metrics, save_path):
    """Create and save visualization of results."""
    fig, axes = plt.subplots(len(predictions) + 1, 5, figsize=(20, 3*(len(predictions) + 1)))
    plt.subplots_adjust(hspace=0.3)   
    
     
    axes[0, 0].imshow(tensor_to_image(frame1))
    axes[0, 0].set_title('Frame 1')
    axes[0, 1].imshow(tensor_to_image(frame2_gt))
    axes[0, 1].set_title('Ground Truth')
    axes[0, 2].imshow(tensor_to_image(frame3))
    axes[0, 2].set_title('Frame 3')
    axes[0, 3].axis('off')
    axes[0, 4].axis('off')
    
     
    for i, (model_name, pred) in enumerate(predictions.items(), 1):
        axes[i, 0].imshow(tensor_to_image(frame1))
        axes[i, 0].set_title('Frame 1')
        
        axes[i, 1].imshow(tensor_to_image(pred))
        axes[i, 1].set_title(f'{model_name}\nPSNR: {metrics[model_name]["PSNR"]:.2f}, SSIM: {metrics[model_name]["SSIM"]:.2f}\nTime: {metrics[model_name]["time"]*1000:.1f}ms')
        
        axes[i, 2].imshow(tensor_to_image(frame3))
        axes[i, 2].set_title('Frame 3')
        
         
        diff = torch.abs(pred - frame2_gt)
        diff = diff[0].mean(dim=0).numpy()
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        axes[i, 3].imshow(diff, cmap='hot')
        axes[i, 3].set_title('Error Map')
        
         
        comparison = torch.cat([frame2_gt, pred], dim=3)
        axes[i, 4].imshow(tensor_to_image(comparison))
        axes[i, 4].set_title('GT vs Predicted')
    
     
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
     
    models = {
        'V1': ModelV1().to(device),
        'V2': ModelV2().to(device),
        'V3': ModelV3().to(device)
    }
    
     
    models['V1'].load_state_dict(torch.load('./v1/best_model_3epochs.pth', weights_only=True)['model_state_dict'])
    models['V2'].load_state_dict(torch.load('./combined_loss_model.pth', weights_only=True)['model_state_dict'])
    models['V3'].load_state_dict(torch.load('./motion_aware_model.pth', weights_only=True)['model_state_dict'])
    
     
    for model in models.values():
        model.eval()
    
     
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
     
    test_folders = ['test/0110', 'test/0728']
    
    for folder in test_folders:
        print(f"\nProcessing sequence in {folder}")
        
         
        frame1 = load_image(os.path.join(folder, 'im1.png'), transform)
        frame2_gt = load_image(os.path.join(folder, 'im2.png'), transform)
        frame3 = load_image(os.path.join(folder, 'im3.png'), transform)
        
         
        frame1 = frame1.unsqueeze(0)
        frame2_gt = frame2_gt.unsqueeze(0)
        frame3 = frame3.unsqueeze(0)
        
         
        predictions = {}
        metrics = {}
        
        for name, model in models.items():
            print(f"Testing model {name}")
            pred, model_metrics = test_sequence(model, frame1, frame3, frame2_gt, device)
            predictions[name] = pred
            metrics[name] = model_metrics
            print(f"Model {name} - PSNR: {model_metrics['PSNR']:.2f}, SSIM: {model_metrics['SSIM']:.2f}, Time: {model_metrics['time']*1000:.1f}ms")
        
         
        save_path = os.path.join(folder, 'comparison.png')
        visualize_results(frame1, frame2_gt, frame3, predictions, metrics, save_path)
        print(f"Results saved to {save_path}")

if __name__ == '__main__':
    main()