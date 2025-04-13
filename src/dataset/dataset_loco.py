import os
from pathlib import Path
from typing import *

import cv2
import numpy as np
from PIL import Image

import torch
import torch.utils
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode
from mask_sampler import BlockMaskingSampler, RandomMaskingSampler, ObjectMaskingSampler, DivisionMaskingSampler, KMeansMaskingSampler, CheckerboardMaskingSampler, \
    MultiBlockMaskingSampler


IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD = [0.229, 0.224, 0.225]

LOCO_CLASSES = {
    "bottle": "juice_bottle",
    "box": "breakfast_box",
    "cable": "splicing_connectors",
    "bag": "screw_bag",
    "pins": "pushpins"
}
LOCO_CLASS_TO_ID = {
    "screw_bag": 0,
    "juice_bottle": 1,
    "splicing_connectors": 2,
    "breakfast_box": 3,
    "pushpins": 4
}

# Normalization for ImageNet
class ImageNetTransforms():
    def __init__(self, input_res: int):
        
        self.mean = torch.Tensor(IMNET_MEAN).view(1, 3, 1, 1)
        self.std = torch.Tensor(IMNET_STD).view(1, 3, 1, 1)
        
        self.img_transform = transforms.Compose([
            transforms.Resize((input_res, input_res)),
            transforms.ToTensor(),
            transforms.Normalize(IMNET_MEAN, IMNET_STD)
        ])
    
    def __call__(self, img: Image) -> torch.Tensor:
        return self.img_transform(img)
    
    def inverse_affine(self, img: torch.Tensor) -> torch.Tensor:
        img = img.to(self.std.device)
        return img * self.std + self.mean
    
class DataAugmentationForMIM(object):
    def __init__(self, args, split):
        
        self.patch_transform = transforms.Compose([
            transforms.Resize((args.tokenizer_input_size, args.tokenizer_input_size)),
            transforms.ToTensor(),
        ])
        self.visual_token_transform = transforms.Compose([
            transforms.Resize((args.input_res, args.input_res)),
            transforms.ToTensor(),
        ])
        if args.masking == "block_random":
            self.masked_position_generator = BlockMaskingSampler(
                args.window_size, num_masking_patches=args.num_mask_patches,
                max_num_patches=args.max_mask_patches_per_block,
                min_num_patches=args.min_mask_patches_per_block,
            )
        elif args.masking == "random":
            self.masked_position_generator = RandomMaskingSampler(
                args.window_size, num_masking_patches=args.num_mask_patches
            )
        elif args.masking == "object":
            self.masked_position_generator = ObjectMaskingSampler(
                args.window_size, args.patch_size, num_objects=args.num_objects, mask_dir=args.mask_dir, \
                    category=args.category
            )
        elif args.masking == "division":
            div_config = {
                "block_num": args.block_num,
                "slice_num": args.slice_num,
                "slice_type": args.slice_type,
                "random_slice_type": args.random_slice_type,
            }
            self.masked_position_generator = DivisionMaskingSampler(
                args.window_size, div_type=args.div_type, div_config=div_config
            )
        elif args.masking == "checkerboard":
            self.masked_position_generator = CheckerboardMaskingSampler(
                args.window_size
            )
        elif args.masking == "kmeans":
            self.masked_position_generator = KMeansMaskingSampler(
                args.window_size, num_masking_patches=args.num_mask_patches, num_clusters=30
            )
        elif args.masking == "multi_block":
            self.masked_position_generator = MultiBlockMaskingSampler(
                args.window_size, num_blocks=args.num_blocks
            )
                
    def __call__(self, image, img_path: str = None):
        return self.patch_transform(image), self.visual_token_transform(image), self.masked_position_generator(image, img_path)

class MVTecLOCO(Dataset):
    def __init__(self, 
        data_root: str, 
        category: str, 
        input_res: int, 
        split: str, 
        custom_transforms: Optional[transforms.Compose] = None,
        is_mask=False, 
        color='rgb',
        cls_label=False
    ):
        """Dataset for MVTec LOCO.
        Args:
            data_root: Root directory of MVTecLOCO dataset. It should contain the data directories for each class under this directory.
            category: Class name. Ex. 'hazelnut'
            input_res: Input resolution of the model.
            split: 'train' or 'test'
            is_mask: If True, return the mask image as the target. Otherwise, return the label.
            color: rgb or grayscale
        """
        self.data_root = data_root
        self.category = category
        self.input_res = input_res
        self.split = split
        self.custom_transforms = custom_transforms
        self.is_mask = is_mask
        self.color = color
        self.cls_label = cls_label
        
        assert Path(self.data_root).exists(), f"Path {self.data_root} does not exist"
        assert self.split == 'train' or self.split == 'val' or self.split == 'test'
        
        # # load files from the dataset
        self.img_files = self.get_files()
        if self.split == 'test':
            self.mask_transform = transforms.Compose(
                [
                    transforms.Resize(input_res),
                    transforms.ToTensor(),
                ]
            )

            self.labels = []
            for file in self.img_files:
                status = str(file).split(os.path.sep)[-2]
                if status == 'good':
                    self.labels.append(0)
                else:
                    self.labels.append(1)
    
    def __getitem__(self, index):
        inputs = {}
        
        img_file = self.img_files[index]
        cls_label = str(img_file).split("/")[-4]
        img = Image.open(img_file)
        
        inputs["clsnames"] = cls_label
        inputs["filenames"] = str(img_file)

        if self.color == 'gray':
            img = img.convert('L')
        
        # sample = self.custom_transforms(img, img_file)
        sample = self.custom_transforms(img)
        
        if self.split == 'train' or self.split == 'val':
            inputs["samples"] = sample
            return inputs
        else:
            if not self.is_mask:
                inputs["samples"] = sample
                inputs["labels"] = self.labels[index]
                if "good" in str(img_file):
                    inputs["anom_type"] = "good"
                elif "logical" in str(img_file):
                    inputs["anom_type"] = "log"
                elif "structural" in str(img_file):
                    inputs["anom_type"] = "str"
                return inputs
            else:
                raise NotImplementedError
    
    def __len__(self):
        return len(self.img_files)
    
    def get_files(self):
        if self.split == 'train':
            files = sorted(Path(os.path.join(self.data_root, self.category, 'train', 'good')).glob('*.png'))
        elif self.split == 'val':
            files = sorted(Path(os.path.join(self.data_root, self.category, 'validation', 'good')).glob('*.png'))
        elif self.split == 'test':
            normal_img_files = sorted(Path(os.path.join(self.data_root, self.category, 'test', 'good')).glob('*.png'))
            logical_img_files = sorted(Path(os.path.join(self.data_root, self.category, 'test', 'logical_anomalies')).glob('*.png'))
            struct_img_files = sorted(Path(os.path.join(self.data_root, self.category, 'test', 'structural_anomalies')).glob('*.png'))
            files = normal_img_files + logical_img_files + struct_img_files
            
        return files

def build_loco_dataset(args, split: str = 'train'):
    transform = DataAugmentationForMIM(args, split)
    if not args.category == "multi":
        dataset = MVTecLOCO(
            data_root=args.data_root,
            category=LOCO_CLASSES[args.category],
            input_res=args.input_res,
            split=split,
            custom_transforms=transform,
            is_mask=args.is_mask,
            cls_label=args.tokenizer == "hvq"
        )
    else:
        dataset = torch.utils.data.ConcatDataset([
            MVTecLOCO(
                data_root=args.data_root,
                category=LOCO_CLASSES[category],
                input_res=args.input_res,
                split=split,
                custom_transforms=transform,
                is_mask=args.is_mask,
                cls_label=args.tokenizer == "hvq"
            )
            for category in LOCO_CLASSES
        ])
    return dataset
    
def build_loco_dataloader(
    args, 
    dataset,
    training
) -> DataLoader:
    
    # build dataloader
    if training:
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True,
        )
    else:
        return DataLoader(
            dataset, 
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False
        )