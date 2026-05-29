# mmdet/datasets/pipelines/bbox_clipper.py
import numpy as np
from mmdet.datasets.builder import PIPELINES  # ← 改用絕對路徑


@PIPELINES.register_module()
class BboxClipper:
    def __call__(self, results):
        if "gt_bboxes" not in results:
            return results
        h, w = results["img"].shape[:2]
        bboxes = results["gt_bboxes"]
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w - 1)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h - 1)
        keep = (bboxes[:, 2] - bboxes[:, 0] > 0) & (bboxes[:, 3] - bboxes[:, 1] > 0)
        results["gt_bboxes"] = bboxes[keep]
        results["gt_labels"] = results["gt_labels"][keep]
        if "gt_masks" in results:
            results["gt_masks"] = results["gt_masks"][keep]
        return results

    def __repr__(self):
        return self.__class__.__name__
