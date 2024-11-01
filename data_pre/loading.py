import cv2
import os

import torch

from data_pre.Basetransforms import BaseTransform
import numpy as np
from typing import Optional
from PIL import Image

from models.structures import get_box_type
from task_modules.bbox.horizontal_boxes import HorizontalBoxes


class LoadImageFromFile(BaseTransform):
    """Load an image from file.

    Required Keys:
    - img_path

    Modified Keys:
    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for cv2 imread.
            Defaults to 'color'.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 ignore_empty: bool = False) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from the dataset.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        filename = results['img_path']

        # Check if file exists
        if not os.path.exists(filename):
            if self.ignore_empty:
                return None
            else:
                raise FileNotFoundError(f"Image file {filename} not found.")

        # Load the image using cv2
        try:
            if self.color_type == 'color':
                img = cv2.imread(filename, cv2.IMREAD_COLOR)
            elif self.color_type == 'grayscale':
                img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            elif self.color_type == 'unchanged':
                img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            else:
                raise ValueError(f"Unsupported color type: {self.color_type}")

            # Ensure the image is loaded properly
            if img is None:
                raise ValueError(f"Failed to load image: {filename}")

            if self.to_float32:
                img = img.astype(np.float32)

            # Store image information
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['ori_shape'] = img.shape[:2]
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"ignore_empty={self.ignore_empty}, "
                f"to_float32={self.to_float32}, "
                f"color_type='{self.color_type}')")

class LoadAnnotations(BaseTransform):

    def __init__(
            self,
            with_bbox: bool = True,
            with_mask: bool = False,
            poly2mask: bool = True,
            box_type: str = 'hbox',
            reduce_zero_label: bool = False,
            ignore_index: int = 255) -> None:
        super().__init__()
        self.with_bbox = with_bbox
        self.with_label = True
        self.with_seg = False
        self.with_keypoints = False
        self.imdecode_backend = 'cv2'
        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        self.with_mask = with_mask
        self.poly2mask = poly2mask
        self.box_type = box_type
        self.reduce_zero_label = reduce_zero_label
        self.ignore_index = ignore_index

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            gt_bboxes.append(instance['bbox'])
            gt_ignore_flags.append(instance['ignore_flag'])
        if self.box_type is None:
            results['gt_bboxes'] = np.array(
                gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            results['gt_bboxes'] = HorizontalBoxes(gt_bboxes, dtype=torch.float32)
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations."""
        gt_bboxes_labels = []
        for instance in results.get('instances', []):
            gt_bboxes_labels.append(instance['bbox_label'])
        results['gt_bboxes_labels'] = np.array(
            gt_bboxes_labels, dtype=np.int64)

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations."""
        if 'seg_map_path' in results:
            seg_map_path = results['seg_map_path']
            if self.imdecode_backend == 'pil':
                seg_map = np.array(Image.open(seg_map_path), dtype=np.uint8)
            else:
                raise ValueError(f"Unsupported imdecode backend: {self.imdecode_backend}")

            results['gt_seg_map'] = seg_map

    def _load_kps(self, results: dict) -> None:
        """Private function to load keypoints annotations."""
        gt_keypoints = [instance['keypoints'] for instance in results['instances']]
        results['gt_keypoints'] = np.array(gt_keypoints, dtype=np.float32).reshape((len(gt_keypoints), -1, 3))

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations."""
        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_seg:
            self._load_seg_map(results)
        if self.with_keypoints:
            self._load_kps(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'with_keypoints={self.with_keypoints}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str