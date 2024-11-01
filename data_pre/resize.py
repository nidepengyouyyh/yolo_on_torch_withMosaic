from data_pre.Basetransforms import BaseTransform
import cv2
from PIL import Image
import numpy as np
from typing import Optional, Tuple, Union, List
import numbers

def impad(img: np.ndarray,
          *,
          shape: Optional[Tuple[int, int]] = None,
          padding: Union[int, tuple, None] = None,
          pad_val: Union[float, List] = 0,
          padding_mode: str = 'constant') -> np.ndarray:
    """Pad the given image to a certain shape or pad on all sides with
    specified padding mode and padding value.

    Args:
        img (ndarray): Image to be padded.
        shape (tuple[int]): Expected padding shape (h, w). Default: None.
        padding (int or tuple[int]): Padding on each border. If a single int is
            provided this is used to pad all borders. If tuple of length 2 is
            provided this is the padding on left/right and top/bottom
            respectively. If a tuple of length 4 is provided this is the
            padding for the left, top, right and bottom borders respectively.
            Default: None. Note that `shape` and `padding` can not be both
            set.
        pad_val (Number | Sequence[Number]): Values to be filled in padding
            areas when padding_mode is 'constant'. Default: 0.
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Default: constant.

            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        ndarray: The padded image.
    """

    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        width = max(shape[1] - img.shape[1], 0)
        height = max(shape[0] - img.shape[0], 0)
        padding = (0, 0, width, height)

    # check pad_val
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError('pad_val must be a int or a tuple. '
                        f'But received {type(pad_val)}')

    # check padding
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                         f'But received {padding}')

    # check padding mode
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

    border_type = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }
    img = cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        border_type[padding_mode],
        value=pad_val)

    return img

def imresize(
    img: np.ndarray,
    size: Tuple[int, int],
    return_scale: bool = False,
    interpolation: str = 'bilinear',
    backend: Optional[str] = 'cv2'
) -> Union[Tuple[np.ndarray, float, float], np.ndarray]:
    """Resize image to a given size without mmcv dependency.

    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`. Default is `cv2`.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
        `resized_img`.
    """
    h, w = img.shape[:2]

    if backend == 'pillow':
        assert img.dtype == np.uint8, 'Pillow backend only supports uint8 type'
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(size, Image.Resampling.BILINEAR if interpolation == 'bilinear' else Image.Resampling.NEAREST)
        resized_img = np.array(pil_image)
    else:
        # Map interpolation methods to OpenCV values.
        cv2_interp_codes = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'area': cv2.INTER_AREA,
            'lanczos': cv2.INTER_LANCZOS4
        }
        resized_img = cv2.resize(img, size, interpolation=cv2_interp_codes.get(interpolation, cv2.INTER_LINEAR))

    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale

def scale_size(size, scale_factor):
    """Calculate new size with a given scale factor."""
    w, h = size
    w_new = int(w * scale_factor[0])
    h_new = int(h * scale_factor[1])
    return (w_new, h_new)

class Resize(BaseTransform):

    def __init__(self,
                 scale: Optional[Union[int, Tuple[int, int]]] = None,
                 scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
                 keep_ratio: bool = False,
                 clip_object_border: bool = True,
                 backend: str = 'cv2',
                 interpolation='bilinear') -> None:
        assert scale is not None or scale_factor is not None, (
            '`scale` and'
            '`scale_factor` cannot both be `None`')

        if scale is None:
            self.scale = None
        else:
            self.scale = (scale, scale) if isinstance(scale, int) else scale

        self.interpolation = interpolation
        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border
        self.backend = backend

        if scale_factor is None:
            self.scale_factor = None
        elif isinstance(scale_factor, float):
            self.scale_factor = (scale_factor, scale_factor)
        elif isinstance(scale_factor, tuple):
            assert len(scale_factor) == 2
            self.scale_factor = scale_factor
        else:
            raise TypeError(
                f'Expected scale_factor to be float or tuple(float), but got {type(scale_factor)}')

    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""

        if results.get('img', None) is not None:
            img = results['img']
            original_shape = img.shape[:2]

            if self.keep_ratio:
                scale_factor = self._calculate_scale_factor(original_shape, results['scale'])
                new_size = (int(original_shape[1] * scale_factor[1]), int(original_shape[0] * scale_factor[0]))
            else:
                new_size = results['scale']

            resized_img = cv2.resize(img, new_size, interpolation=self._get_interpolation())
            results['img'] = resized_img
            results['img_shape'] = resized_img.shape[:2]
            results['scale_factor'] = (new_size[0] / original_shape[1], new_size[1] / original_shape[0])
            results['keep_ratio'] = self.keep_ratio

    def _calculate_scale_factor(self, original_shape, target_size):
        """Calculate scale factor while keeping aspect ratio."""
        orig_h, orig_w = original_shape
        target_w, target_h = target_size
        scale_factor = min(target_w / orig_w, target_h / orig_h)
        return scale_factor, scale_factor

    def _get_interpolation(self):
        """Map interpolation string to OpenCV interpolation flag."""
        mapping = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'area': cv2.INTER_AREA,
            'lanczos': cv2.INTER_LANCZOS4
        }
        return mapping.get(self.interpolation, cv2.INTER_LINEAR)

    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'].rescale_(results['scale_factor'])
            if self.clip_object_border:
                results['gt_bboxes'].clip_(results['img_shape'])

    # def _resize_seg(self, results: dict) -> None:
    #     """Resize semantic segmentation map with ``results['scale']``."""
    #     if results.get('gt_seg_map', None) is not None:
    #         if self.keep_ratio:
    #             gt_seg = mmcv.imrescale(
    #                 results['gt_seg_map'],
    #                 results['scale'],
    #                 interpolation='nearest',
    #                 backend=self.backend)
    #         else:
    #             gt_seg = mmcv.imresize(
    #                 results['gt_seg_map'],
    #                 results['scale'],
    #                 interpolation='nearest',
    #                 backend=self.backend)
    #         results['gt_seg_map'] = gt_seg

    def _resize_keypoints(self, results: dict) -> None:
        """Resize keypoints with ``results['scale_factor']``."""
        if results.get('gt_keypoints', None) is not None:
            keypoints = results['gt_keypoints']
            keypoints[:, :, :2] = keypoints[:, :, :2] * np.array(results['scale_factor'])

            if self.clip_object_border:
                keypoints[:, :, 0] = np.clip(keypoints[:, :, 0], 0, results['img_shape'][1])
                keypoints[:, :, 1] = np.clip(keypoints[:, :, 1], 0, results['img_shape'][0])
            results['gt_keypoints'] = keypoints


    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, and segmentation maps."""
        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = scale_size(img_shape[::-1], self.scale_factor)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        # self._resize_seg(results)
        self._record_homography_matrix(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'scale_factor={self.scale_factor}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'clip_object_border={self.clip_object_border}), '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str

    def _resize_masks(self, results: dict) -> None:
        """Resize masks with ``results['scale']``"""
        if results.get('gt_masks', None) is not None:
            if self.keep_ratio:
                results['gt_masks'] = results['gt_masks'].rescale(
                    results['scale'])
            else:
                results['gt_masks'] = results['gt_masks'].resize(
                    results['img_shape'])

    def _record_homography_matrix(self, results: dict) -> None:
        """Record the homography matrix for the Resize."""
        w_scale, h_scale = results['scale_factor']
        homography_matrix = np.array(
            [[w_scale, 0, 0], [0, h_scale, 0], [0, 0, 1]], dtype=np.float32)
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

class YOLOv5KeepRatioResize(Resize):
    def __init__(self,
                 scale: Union[int, Tuple[int, int]],
                 keep_ratio: bool = True,
                 **kwargs):
        assert keep_ratio is True
        super().__init__(scale=scale, keep_ratio=True, **kwargs)

    @staticmethod
    def _get_rescale_ratio(old_size: Tuple[int, int],
                           scale: Union[float, Tuple[int]]) -> float:
        """Calculate the ratio for rescaling.

        Args:
            old_size (tuple[int]): The old size (w, h) of image.
            scale (float | tuple[int]): The scaling factor or maximum size.
                If it is a float number, then the image will be rescaled by
                this factor, else if it is a tuple of 2 integers, then
                the image will be rescaled as large as possible within
                the scale.

        Returns:
            float: The resize ratio.
        """
        w, h = old_size
        if isinstance(scale, (float, int)):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
            scale_factor = scale
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            scale_factor = min(max_long_edge / max(h, w),
                               max_short_edge / min(h, w))
        else:
            raise TypeError('Scale must be a number or tuple of int, '
                            f'but got {type(scale)}')

        return scale_factor

    def _resize_img(self, results: dict):
        """Resize images with ``results['scale']``."""
        assert self.keep_ratio is True

        if results.get('img', None) is not None:
            image = results['img']
            original_h, original_w = image.shape[:2]
            ratio = self._get_rescale_ratio((original_h, original_w),
                                            self.scale)

            if ratio != 1:
                # resize image according to the shape
                # NOTE: We are currently testing on COCO that modifying
                # this code will not affect the results.
                # If you find that it has an effect on your results,
                # please feel free to contact us.
                image = imresize(
                    img=image,
                    size=(int(original_w * ratio), int(original_h * ratio)),
                    interpolation='area' if ratio < 1 else 'bilinear',
                    backend=self.backend)

            resized_h, resized_w = image.shape[:2]
            scale_ratio_h = resized_h / original_h
            scale_ratio_w = resized_w / original_w
            scale_factor = (scale_ratio_w, scale_ratio_h)

            results['img'] = image
            results['img_shape'] = image.shape[:2]
            results['scale_factor'] = scale_factor

class LetterResize(Resize):
    """Resize and pad image while meeting stride-multiple constraints.

    Required Keys:

    - img (np.uint8)
    - batch_shape (np.int64) (optional)

    Modified Keys:

    - img (np.uint8)
    - img_shape (tuple)
    - gt_bboxes (optional)

    Added Keys:
    - pad_param (np.float32)

    Args:
        scale (Union[int, Tuple[int, int]]): Images scales for resizing.
        pad_val (dict): Padding value. Defaults to dict(img=0, seg=255).
        use_mini_pad (bool): Whether using minimum rectangle padding.
            Defaults to True
        stretch_only (bool): Whether stretch to the specified size directly.
            Defaults to False
        allow_scale_up (bool): Allow scale up when ratio > 1. Defaults to True
        half_pad_param (bool): If set to True, left and right pad_param will
            be given by dividing padding_h by 2. If set to False, pad_param is
            in int format. We recommend setting this to False for object
            detection tasks, and True for instance segmentation tasks.
            Default to False.
    """

    def __init__(self,
                 scale: Union[int, Tuple[int, int]],
                 pad_val: dict = dict(img=0, mask=0, seg=255),
                 use_mini_pad: bool = False,
                 stretch_only: bool = False,
                 allow_scale_up: bool = True,
                 half_pad_param: bool = False,
                 **kwargs):
        super().__init__(scale=scale, keep_ratio=True, **kwargs)

        self.pad_val = pad_val
        if isinstance(pad_val, (int, float)):
            pad_val = dict(img=pad_val, seg=255)
        assert isinstance(
            pad_val, dict), f'pad_val must be dict, but got {type(pad_val)}'

        self.use_mini_pad = use_mini_pad
        self.stretch_only = stretch_only
        self.allow_scale_up = allow_scale_up
        self.half_pad_param = half_pad_param

    def _resize_img(self, results: dict):
        """Resize images with ``results['scale']``."""
        image = results.get('img', None)
        if image is None:
            return

        # Use batch_shape if a batch_shape policy is configured
        if 'batch_shape' in results:
            scale = tuple(results['batch_shape'])  # hw
        else:
            scale = self.scale[::-1]  # wh -> hw

        image_shape = image.shape[:2]  # height, width

        # Scale ratio (new / old)
        ratio = min(scale[0] / image_shape[0], scale[1] / image_shape[1])

        # only scale down, do not scale up (for better test mAP)
        if not self.allow_scale_up:
            ratio = min(ratio, 1.0)

        ratio = [ratio, ratio]  # float -> (float, float) for (height, width)

        # compute the best size of the image
        no_pad_shape = (int(round(image_shape[0] * ratio[0])),
                        int(round(image_shape[1] * ratio[1])))

        # padding height & width
        padding_h, padding_w = [
            scale[0] - no_pad_shape[0], scale[1] - no_pad_shape[1]
        ]
        if self.use_mini_pad:
            # minimum rectangle padding
            padding_w, padding_h = np.mod(padding_w, 32), np.mod(padding_h, 32)

        elif self.stretch_only:
            # stretch to the specified size directly
            padding_h, padding_w = 0.0, 0.0
            no_pad_shape = (scale[0], scale[1])
            ratio = [scale[0] / image_shape[0],
                     scale[1] / image_shape[1]]  # height, width ratios

        if image_shape != no_pad_shape:
            # compare with no resize and padding size
            image = imresize(
                image, (no_pad_shape[1], no_pad_shape[0]),
                interpolation=self.interpolation,
                backend=self.backend)

        scale_factor = (no_pad_shape[1] / image_shape[1],
                        no_pad_shape[0] / image_shape[0])

        if 'scale_factor' in results:
            results['scale_factor_origin'] = results['scale_factor']
        results['scale_factor'] = scale_factor

        # padding
        top_padding, left_padding = int(round(padding_h // 2 - 0.1)), int(
            round(padding_w // 2 - 0.1))
        bottom_padding = padding_h - top_padding
        right_padding = padding_w - left_padding

        padding_list = [
            top_padding, bottom_padding, left_padding, right_padding
        ]
        if top_padding != 0 or bottom_padding != 0 or \
                left_padding != 0 or right_padding != 0:

            pad_val = self.pad_val.get('img', 0)
            if isinstance(pad_val, int) and image.ndim == 3:
                pad_val = tuple(pad_val for _ in range(image.shape[2]))

            image = impad(
                img=image,
                padding=(padding_list[2], padding_list[0], padding_list[3],
                         padding_list[1]),
                pad_val=pad_val,
                padding_mode='constant')

        results['img'] = image
        results['img_shape'] = image.shape
        if 'pad_param' in results:
            results['pad_param_origin'] = results['pad_param'] * \
                                          np.repeat(ratio, 2)

        if self.half_pad_param:
            results['pad_param'] = np.array(
                [padding_h / 2, padding_h / 2, padding_w / 2, padding_w / 2],
                dtype=np.float32)
        else:
            # We found in object detection, using padding list with
            # int type can get higher mAP.
            results['pad_param'] = np.array(padding_list, dtype=np.float32)

    # def _resize_masks(self, results: dict):
    #     """Resize masks with ``results['scale']``"""
    #     if results.get('gt_masks', None) is None:
    #         return
    #
    #     gt_masks = results['gt_masks']
    #     assert isinstance(
    #         gt_masks, PolygonMasks
    #     ), f'Only supports PolygonMasks, but got {type(gt_masks)}'
    #
    #     # resize the gt_masks
    #     gt_mask_h = results['gt_masks'].height * results['scale_factor'][1]
    #     gt_mask_w = results['gt_masks'].width * results['scale_factor'][0]
    #     gt_masks = results['gt_masks'].resize(
    #         (int(round(gt_mask_h)), int(round(gt_mask_w))))
    #
    #     top_padding, _, left_padding, _ = results['pad_param']
    #     if int(left_padding) != 0:
    #         gt_masks = gt_masks.translate(
    #             out_shape=results['img_shape'][:2],
    #             offset=int(left_padding),
    #             direction='horizontal')
    #     if int(top_padding) != 0:
    #         gt_masks = gt_masks.translate(
    #             out_shape=results['img_shape'][:2],
    #             offset=int(top_padding),
    #             direction='vertical')
    #     results['gt_masks'] = gt_masks

    def _resize_bboxes(self, results: dict):
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is None:
            return
        results['gt_bboxes'].rescale_(results['scale_factor'])

        if len(results['pad_param']) != 4:
            return
        results['gt_bboxes'].translate_(
            (results['pad_param'][2], results['pad_param'][0]))

        if self.clip_object_border:
            results['gt_bboxes'].clip_(results['img_shape'])

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        if 'scale_factor_origin' in results:
            scale_factor_origin = results.pop('scale_factor_origin')
            results['scale_factor'] = (results['scale_factor'][0] *
                                       scale_factor_origin[0],
                                       results['scale_factor'][1] *
                                       scale_factor_origin[1])
        if 'pad_param_origin' in results:
            pad_param_origin = results.pop('pad_param_origin')
            results['pad_param'] += pad_param_origin
        return results