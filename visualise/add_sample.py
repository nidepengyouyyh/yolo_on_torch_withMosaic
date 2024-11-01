# import cv2
# import numpy as np
# from typing import Optional
#
# def add_datasample(
#         name: str,
#         image: np.ndarray,
#         data_sample=None,
#         draw_gt: bool = True,
#         draw_pred: bool = True,
#         show: bool = False,
#         wait_time: float = 0,
#         out_file: Optional[str] = None,
#         pred_score_thr: float = 0.3,
#         step: int = 0) -> None:
#     """Draw datasample and save to all backends.
#
#     - If GT and prediction are plotted at the same time, they are
#     displayed in a stitched image where the left image is the
#     ground truth and the right image is the prediction.
#     - If ``show`` is True, all storage backends are ignored, and
#     the images will be displayed in a local window.
#     - If ``out_file`` is specified, the drawn image will be
#     saved to ``out_file``. It is usually used when the display
#     is not available.
#
#     Args:
#         name (str): The image identifier.
#         image (np.ndarray): The image to draw.
#         data_sample (:obj:`DetDataSample`, optional): A data
#             sample that contains annotations and predictions.
#             Defaults to None.
#         draw_gt (bool): Whether to draw GT DetDataSample. Default to True.
#         draw_pred (bool): Whether to draw Prediction DetDataSample.
#             Defaults to True.
#         show (bool): Whether to display the drawn image. Default to False.
#         wait_time (float): The interval of show (s). Defaults to 0.
#         out_file (str): Path to output file. Defaults to None.
#         pred_score_thr (float): The threshold to visualize the bboxes
#             and masks. Defaults to 0.3.
#         step (int): Global step value to record. Defaults to 0.
#     """
#     image = image.clip(0, 255).astype(np.uint8)
#     dataset_meta = {}
#     classes = dataset_meta.get('classes', None)
#     palette = dataset_meta.get('palette', None)
#
#     gt_img_data = None
#     pred_img_data = None
#
#     if data_sample is not None:
#         data_sample = data_sample.cpu()
#
#     if draw_gt and data_sample is not None:
#         gt_img_data = image
#         if 'gt_instances' in data_sample:
#             gt_img_data = self._draw_instances(image,
#                                                data_sample.gt_instances,
#                                                classes, palette)
#         if 'gt_sem_seg' in data_sample:
#             gt_img_data = self._draw_sem_seg(gt_img_data,
#                                              data_sample.gt_sem_seg,
#                                              classes, palette)
#
#         if 'gt_panoptic_seg' in data_sample:
#             assert classes is not None, 'class information is ' \
#                                         'not provided when ' \
#                                         'visualizing panoptic ' \
#                                         'segmentation results.'
#             gt_img_data = self._draw_panoptic_seg(
#                 gt_img_data, data_sample.gt_panoptic_seg, classes, palette)
#
#     if draw_pred and data_sample is not None:
#         pred_img_data = image
#         if 'pred_instances' in data_sample:
#             pred_instances = data_sample.pred_instances
#             pred_instances = pred_instances[
#                 pred_instances.scores > pred_score_thr]
#             pred_img_data = self._draw_instances(image, pred_instances,
#                                                  classes, palette)
#
#         if 'pred_sem_seg' in data_sample:
#             pred_img_data = self._draw_sem_seg(pred_img_data,
#                                                data_sample.pred_sem_seg,
#                                                classes, palette)
#
#         if 'pred_panoptic_seg' in data_sample:
#             assert classes is not None, 'class information is ' \
#                                         'not provided when ' \
#                                         'visualizing panoptic ' \
#                                         'segmentation results.'
#             pred_img_data = self._draw_panoptic_seg(
#                 pred_img_data, data_sample.pred_panoptic_seg.numpy(),
#                 classes, palette)
#
#     if gt_img_data is not None and pred_img_data is not None:
#         drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
#     elif gt_img_data is not None:
#         drawn_img = gt_img_data
#     elif pred_img_data is not None:
#         drawn_img = pred_img_data
#     else:
#         # Display the original image directly if nothing is drawn.
#         drawn_img = image
#
#     # It is convenient for users to obtain the drawn image.
#     # For example, the user wants to obtain the drawn image and
#     # save it as a video during video inference.
#     self.set_image(drawn_img)
#
#     if show:
#         self.show(drawn_img, win_name=name, wait_time=wait_time)
#
#     if out_file is not None:
#         # Use OpenCV's imwrite function to save the image
#         cv2.imwrite(out_file, drawn_img[..., ::-1])
#     else:
#         self.add_image(name, drawn_img, step)