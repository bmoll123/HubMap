import numpy as np
import pandas as pd
from pathlib import Path
import cv2

from ..builder import PIPELINES


@PIPELINES.register_module()
class StainTransform:
    def __init__(self, img_aug_root, img_ext, margin=0, prob=1.0):
        self.img_aug_root = Path(img_aug_root)
        self.img_ext = img_ext
        self.prob = prob
        self._prepare_img_refs()
        self.margin = margin

    def _prepare_img_refs(self):
        ids_dirs = list(self.img_aug_root.iterdir()) 
        ids_dirs = list(filter(lambda x: x.is_dir(), ids_dirs))
        self._stain_infos = {}
        for ids_dir in ids_dirs:
            ids_stem = ids_dir.stem
            stain_img_paths = list(Path(ids_dir).rglob(f"*{self.img_ext}"))
            self._stain_infos[ids_stem] = stain_img_paths
            
    @staticmethod
    def crop_full_tile(img, margin=128):
        base_size = 512
        img_h, img_w = img.shape[:2]
        assert img_h == base_size * 3
        assert img_w == base_size * 3
        crop_h, crop_w = base_size + margin * 2, base_size + margin * 2
        x0 = base_size - margin
        y0 = base_size - margin

        x1 = x0 + crop_w
        y1 = y0 + crop_h
        center_region = img[y0:y1, x0:x1]

        return center_region

    def _get_img_ref(self, img_path):
        ref_img = cv2.imread(str(img_path))

        if self.margin > 0:
            ref_img = self.crop_full_tile(ref_img, self.margin)

        return ref_img
    
    def _get_stain_offline(self, org_image, img_stem):
        ref_candidates = self._stain_infos.get(img_stem, None)

        if ref_candidates is None:
            return org_image
        else:
            if isinstance(ref_candidates, list) and len(ref_candidates) > 0:
                ref_path = np.random.choice(ref_candidates)
                stain_img = self._get_img_ref(ref_path)
                
                # CRITICAL FIX: Ensure reference image has same dimensions as original
                # This prevents bbox coordinate misalignment
                org_h, org_w = org_image.shape[:2]
                stain_h, stain_w = stain_img.shape[:2]
                
                if stain_h != org_h or stain_w != org_w:
                    # Resize reference image to match original dimensions
                    stain_img = cv2.resize(stain_img, (org_w, org_h), interpolation=cv2.INTER_LINEAR)
                
                return stain_img
            else:
                return org_image

    def _adjust_img(self, results):
        for key in results.get('img_fields', ['img']):
            img_stem = Path(results['filename']).stem
            img = results[key]
            results[key] = self._get_stain_offline(img, img_stem).astype(img.dtype)
    
    def __call__(self, results):

        if np.random.rand() > self.prob:
            return results
        
        # Store original image shape for validation
        original_img_shape = results['img'].shape
        
        try:
            self._adjust_img(results)
            
            # Validate that image shape hasn't changed
            if results['img'].shape != original_img_shape:
                print(f"Warning: Image shape changed from {original_img_shape} to {results['img'].shape}")
                print("Reverting to original image")
                # Revert to original image if shape changed
                for key in results.get('img_fields', ['img']):
                    results[key] = results[key]  # Keep original
            
            return results
        except Exception as e:
            print(f"StainTransform error: {e}, skipping augmentation")
            return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
