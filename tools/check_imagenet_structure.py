#!/usr/bin/env python3
"""
Simple script to check ImageNet directory structure without external dependencies.
"""

import os

def check_imagenet_structure():
    """Check if ImageNet directory structure is correct."""
    data_root = "/data02/sqy/datasets/ILSVRC"
    
    print("ImageNet Directory Structure Check")
    print("=" * 50)
    
    # Check training directory
    train_dir = os.path.join(data_root, "ILSVRC2012_train", "data")
    print(f"\nTraining Directory: {train_dir}")
    
    if os.path.exists(train_dir):
        print("✓ Training directory exists")
        
        # List synset directories
        synsets = [d for d in os.listdir(train_dir) 
                  if os.path.isdir(os.path.join(train_dir, d)) and d.startswith('n')]
        print(f"✓ Found {len(synsets)} synset directories")
        
        if synsets:
            # Check first few synsets
            print(f"  First 5 synsets: {synsets[:5]}")
            
            # Check contents of first synset
            first_synset_path = os.path.join(train_dir, synsets[0])
            jpeg_files = [f for f in os.listdir(first_synset_path) if f.endswith('.JPEG')]
            print(f"  {synsets[0]} contains {len(jpeg_files)} JPEG files")
            
            if jpeg_files:
                print(f"  Example files: {jpeg_files[:3]}")
            
            # Expected structure validation
            total_images = 0
            checked_synsets = 0
            for synset in synsets[:10]:  # Check first 10 synsets
                synset_path = os.path.join(train_dir, synset)
                jpeg_count = len([f for f in os.listdir(synset_path) if f.endswith('.JPEG')])
                total_images += jpeg_count
                checked_synsets += 1
            
            avg_images_per_synset = total_images / checked_synsets if checked_synsets > 0 else 0
            print(f"  Average images per synset (first 10): {avg_images_per_synset:.1f}")
            
            # ImageNet training set should have 1000 synsets and ~1.28M images total
            if len(synsets) == 1000:
                print("✓ Correct number of synsets (1000)")
            else:
                print(f"⚠ Expected 1000 synsets, found {len(synsets)}")
                
        else:
            print("✗ No synset directories found")
    else:
        print("✗ Training directory does not exist")
    
    # Check validation directory
    val_dir = os.path.join(data_root, "ILSVRC2012_validation", "data")
    print(f"\nValidation Directory: {val_dir}")
    
    if os.path.exists(val_dir):
        print("✓ Validation directory exists")
        
        # Check if validation is organized by synsets or flat
        val_contents = os.listdir(val_dir)
        synset_dirs = [d for d in val_contents if os.path.isdir(os.path.join(val_dir, d)) and d.startswith('n')]
        jpeg_files = [f for f in val_contents if f.endswith('.JPEG')]
        
        if synset_dirs:
            print(f"✓ Validation organized by synsets: {len(synset_dirs)} synsets")
            total_val_images = 0
            for synset in synset_dirs[:5]:  # Check first 5
                synset_path = os.path.join(val_dir, synset)
                count = len([f for f in os.listdir(synset_path) if f.endswith('.JPEG')])
                total_val_images += count
            print(f"  Sample validation images: {total_val_images} (from first 5 synsets)")
        elif jpeg_files:
            print(f"⚠ Validation appears to be flat structure: {len(jpeg_files)} JPEG files")
            print("  Note: You may need to reorganize validation set by synsets")
        else:
            print("✗ No validation images found")
    else:
        print("✗ Validation directory does not exist")
    
    # Check for auxiliary files
    print(f"\nAuxiliary Files:")
    aux_files = [
        ("index_synset.yaml", "Synset index mapping"),
        ("synset_human.txt", "Human-readable synset names"),
        ("imagenet1000_clsidx_to_labels.txt", "Class index to labels"),
    ]
    
    for filename, description in aux_files:
        filepath = os.path.join(data_root, "ILSVRC2012_train", filename)
        if os.path.exists(filepath):
            print(f"✓ {description}: {filepath}")
        else:
            print(f"- {description}: {filepath} (will be downloaded automatically)")
    
    # Summary and recommendations
    print(f"\nSummary and Recommendations:")
    print("=" * 50)
    
    if os.path.exists(train_dir) and synsets and len(synsets) == 1000:
        print("✓ Your ImageNet training structure looks correct!")
        print("  Ready to use with ImageNetTrain class")
    else:
        print("⚠ Training structure needs attention")
    
    if os.path.exists(val_dir):
        if synset_dirs:
            print("✓ Validation structure looks correct!")
        else:
            print("⚠ Validation may need reorganization into synset directories")
    else:
        print("⚠ Validation directory missing")
    
    print(f"\nTo use in your code:")
    print("```python")
    print("from vavae.ldm.data.imagenet import ImageNetTrain, ImageNetValidation")
    print("")
    print("# For training")
    print("train_dataset = ImageNetTrain(")
    print(f"    data_root='{data_root}',")
    print("    config={'size': 256, 'random_crop': True}")
    print(")")
    print("")
    print("# For validation")
    print("val_dataset = ImageNetValidation(")
    print(f"    data_root='{data_root}',")
    print("    config={'size': 256, 'random_crop': False}")
    print(")")
    print("```")

if __name__ == "__main__":
    check_imagenet_structure() 