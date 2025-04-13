
import os
from pathlib import Path

import random
import math
import numpy as np
import cv2
from typing import List, Tuple, Dict, Union, Optional

LOCO_CLASSES = {
    "bottle": "juice_bottle",
    "box": "breakfast_box",
    "cable": "splicing_connectors",
    "bag": "screw_bag",
    "pins": "pushpins"
}
        
class RandomMaskingSampler:
    def __init__(
        self, input_size, num_masking_patches
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.input_size = input_size
        self.num_masking_patches = num_masking_patches  
    
    def get_shape(self):
        return self.input_size
    
    def _mask(self):
        delta = 0
        mask_count = 0
        mask = np.zeros(shape=self.get_shape(), dtype=np.int32)
        while mask_count < self.num_masking_patches:
            mask_x = random.randint(0, self.input_size[0] - 1)
            mask_y = random.randint(0, self.input_size[1] - 1)
            if mask[mask_x, mask_y] == 0:
                mask[mask_x, mask_y] = 1
                mask_count += 1
                delta += 1
                
        assert mask_count == self.num_masking_patches, f"mask: {mask}, mask count {mask_count}"
        return mask
    
    def __call__(self, image, img_path):
        return self._mask()
    
class CheckerboardMaskingSampler:
    def __init__(
        self, input_size, div_list=[2,4,8],  
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.input_size = input_size
        self.div_list = div_list
        
        self.all_masks = self.collate_all_masks()
        
    def get_shape(self):
        return self.input_size
    
    def collate_all_masks(self):
        masks = []
        for div in self.div_list:
            mask = np.zeros(shape=self.get_shape(), dtype=np.int32)
            grid_size = self.input_size[0] // div
            for i in range(div):
                for j in range(div):
                    if (i + j) % 2 == 0:
                        mask[i * grid_size: (i + 1) * grid_size, j * grid_size: (j + 1) * grid_size] = 1
            inv_mask = np.logical_not(mask)
            masks.append(mask.astype(np.int32))
            masks.append(inv_mask.astype(np.int32))
        return masks
    
    def __call__(self, image, img_path):
        mask = random.choice(self.all_masks)
        return mask
            
class BlockMaskingSampler:
    def __init__(
        self, input_size, num_masking_patches, min_num_patches=4, max_num_patches=None,
        min_aspect=0.3, max_aspect=None
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size
        
        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches
        
        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches
        
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
    
    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)
                
                num_masked = mask[top: top + h, left: left + w].sum()  # overlap on the previous mask
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1
                    # delta represents the number of pixels that have been masked w/o overlap
                
                if delta > 0:
                    break
        return delta
    
    def __call__(self, img, img_path):
        mask = np.zeros(shape=self.get_shape(), dtype=np.int32)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.max_num_patches - mask_count
            
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta
        
        if mask_count > self.num_masking_patches:
            delta = mask_count - self.num_masking_patches
            mask_x, mask_y = mask.nonzero()
            to_vis = np.random.choice(mask_x.shape[0], delta, replace=False)
            mask[mask_x[to_vis], mask_y[to_vis]] = 0

        elif mask_count < self.num_masking_patches:
            delta = self.num_masking_patches - mask_count
            mask_x, mask_y = (mask == 0).nonzero()
            to_mask = np.random.choice(mask_x.shape[0], delta, replace=False)
            mask[mask_x[to_mask], mask_y[to_mask]] = 1
        
        assert mask.sum() == self.num_masking_patches, f"mask: {mask}, mask count {mask.sum()}"
        return mask
    
    def iterate_masks_random(self, n_masks=10, overlap_thresh=0.5):
        # sample n_masks masks
        masks = []
        for _ in range(n_masks):
            mask = np.zeros(shape=self.get_shape(), dtype=np.int32)
            mask_count = 0
            while mask_count < self.num_masking_patches:
                max_mask_patches = self.max_num_patches - mask_count
                
                delta = self._mask(mask, max_mask_patches)
                if delta == 0:
                    break
                else:
                    mask_count += delta
            
            if mask_count > self.num_masking_patches:
                delta = mask_count - self.num_masking_patches
                mask_x, mask_y = mask.nonzero()
                to_vis = np.random.choice(mask_x.shape[0], delta, replace=False)
                mask[mask_x[to_vis], mask_y[to_vis]] = 0

            elif mask_count < self.num_masking_patches:
                delta = self.num_masking_patches - mask_count
                mask_x, mask_y = (mask == 0).nonzero()
                to_mask = np.random.choice(mask_x.shape[0], delta, replace=False)
                mask[mask_x[to_mask], mask_y[to_mask]] = 1
            
            assert mask.sum() == self.num_masking_patches, f"mask: {mask}, mask count {mask.sum()}"
            masks.append(mask)
        
        return masks
        

def calculate_overlap(self, mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union

def dilate(mask, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)

def expand_mask(mask, window_size, max_width, max_height, auto=False, rm_unconnected=True, min_width=2, min_height=2):
    if rm_unconnected:
        mask = remove_unconnected_region(mask.astype(np.int8))
    if auto:
        mask_x, mask_y = np.where(mask > 0)
        x1, x2 = mask_x.min(), mask_x.max()
        y1, y2 = mask_y.min(), mask_y.max()
    else:
        mask_x, mask_y = np.where(mask > 0)
        center_x = int(mask_x.mean())
        center_y = int(mask_y.mean())
        width = random.randint(min_width, max_width)
        height = random.randint(min_height, max_height)
        x1 = max(center_x - width // 2, 0)
        x2 = min(center_x + width // 2, window_size)
        y1 = max(center_y - height // 2, 0)
        y2 = min(center_y + height // 2, window_size)
    
    mask = np.zeros((window_size, window_size))
    mask[x1:x2, y1:y2] = 1
    return mask

def remove_unconnected_region(mask):
    mask = mask.copy()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 2:
        largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        mask[labels != largest_label] = 0
    return mask
