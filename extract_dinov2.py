#!/usr/bin/env python3
"""
Script to extract DINOv2 features from ImageNet dataset using multi-GPU inference.
Features are saved as .pth files by replacing .JPEG extension in original image paths.
Uses distributed data loading (not DDP) for multi-GPU feature extraction.

Usage:
    # Single GPU
    python extract_dinov2.py --data_root /path/to/imagenet --split train

    # Multi-GPU
    torchrun --nproc_per_node=4 extract_dinov2.py --data_root /data02/sqy/datasets/ILSVRC --split train

by Jingfeng Yao
from HUST-VL
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'vavae'))

from vavae.ldm.models.foundation_models import aux_foundation_model
from vavae.ldm.data.imagenet import ImageNetTrain, ImageNetValidation

warnings.filterwarnings("ignore")


class ImageNetFeatureDataset(Dataset):
    """
    Custom dataset for feature extraction that returns image paths and preprocessed tensors
    """
    def __init__(self, imagenet_dataset, transform=None):
        self.imagenet_dataset = imagenet_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.imagenet_dataset)
    
    def __getitem__(self, idx):
        # Get the data from ImageNet dataset
        data = self.imagenet_dataset[idx]
        
        # Extract image path and load image
        if isinstance(data, dict):
            image_path = data.get('file_path_', '')
            if 'image' in data:
                # Image is already loaded
                image = data['image']
                # Convert from [-1, 1] back to [0, 1] and then to PIL
                image = (image + 1.0) / 2.0
                image = (image * 255).astype(np.uint8)
                if len(image.shape) == 3 and image.shape[0] == 3:
                    image = image.transpose(1, 2, 0)  # CHW to HWC
                image = Image.fromarray(image)
            else:
                # Load image from path
                image = Image.open(image_path).convert('RGB')
        else:
            # Fallback - assume it's a path
            image_path = str(data)
            image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        # Get relative path for saving
        if hasattr(self.imagenet_dataset, 'relpaths'):
            rel_path = self.imagenet_dataset.relpaths[idx]
        else:
            rel_path = os.path.basename(image_path)
            
        return {
            'image': image,
            'rel_path': rel_path,
            'full_path': image_path
        }


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_dataloader(args, rank, world_size):
    """Create data loader for the specified split"""
    # Create ImageNet dataset
    if args.split == 'train':
        dataset = ImageNetTrain(
            data_root=args.data_root,
            process_images=True,
            config={'size': 224}
        )
    elif args.split == 'val':
        dataset = ImageNetValidation(
            data_root=args.data_root, 
            process_images=True,
            config={'size': 224}
        )
    else:
        raise ValueError(f"Unsupported split: {args.split}")
    
    # Create transform for DINOv2 (standard ImageNet preprocessing)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Wrap in feature extraction dataset
    feature_dataset = ImageNetFeatureDataset(dataset, transform=transform)
    
    # Create distributed sampler if using multiple GPUs
    if world_size > 1:
        sampler = DistributedSampler(
            feature_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        sampler = None
    
    # Create data loader
    dataloader = DataLoader(
        feature_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader


def extract_features(args):
    """Main feature extraction function"""
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Print info only on rank 0
    if rank == 0:
        print(f"Starting DINOv2 feature extraction...")
        print(f"World size: {world_size}, Local rank: {local_rank}")
        print(f"Data root: {args.data_root}")
        print(f"Split: {args.split}")
        print(f"Batch size: {args.batch_size}")
    
    # Load DINOv2 model
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    model = aux_foundation_model('dinov2')
    model.to(device)
    model.eval()  # Use eval mode for inference
    
    # Set requires_grad=False for all parameters (inference only)
    for param in model.parameters():
        param.requires_grad = False
    
    # Create data loader
    dataloader = create_dataloader(args, rank, world_size)
    
    # Extract features
    total_processed = 0
    
    if rank == 0:
        pbar = tqdm(total=len(dataloader), desc="Extracting features")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            rel_paths = batch['rel_path']
            full_paths = batch['full_path']
            
            # Extract features
            features = model(images)
            
            # Save features for each image in the batch
            for i, (rel_path, full_path) in enumerate(zip(rel_paths, full_paths)):
                # Replace .JPEG/.jpeg with .pth in the original full path
                feature_path = full_path.replace('.JPEG', '.pth').replace('.jpeg', '.pth')
                full_output_path = Path(feature_path)
                
                # Create subdirectory if needed
                full_output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Extract single image feature
                single_feature = features[i].cpu()
                
                # Save feature
                torch.save({
                    'features': single_feature,
                    'shape': single_feature.shape,
                    'original_path': rel_path,
                    'model_type': 'dinov2'
                }, full_output_path)
            
            total_processed += len(rel_paths)
            
            if rank == 0:
                pbar.update(1)
                if batch_idx % 100 == 0:
                    pbar.set_postfix({
                        'processed': total_processed,
                        'batch': f"{batch_idx+1}/{len(dataloader)}"
                    })
    
    if rank == 0:
        pbar.close()
        print(f"Feature extraction completed! Processed {total_processed} images.")
        print(f"Features saved alongside original images (replaced .JPEG with .pth)")
    
    # Cleanup
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='Extract DINOv2 features from ImageNet')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of ImageNet dataset')
    parser.add_argument('--split', type=str, choices=['train', 'val'], default='train',
                        help='Dataset split to process')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data_root):
        raise ValueError(f"Data root does not exist: {args.data_root}")
    
    # Run feature extraction
    extract_features(args)


if __name__ == '__main__':
    main()
