import cv2
import numpy as np


def sliding_window_segmentation_in_memory(image_path, labels, window_size=(128, 128), step_size=127):
    """
    对输入的图像和标签应用滑动窗口分割，并同步调整标签坐标。

    参数:
        image (ndarray): 输入图像，形状为 (高度, 宽度, 通道数)。
        labels (list of lists): 标签列表，每个标签为 [x1, y1, x2, y2, x3, y3, x4, y4, class_name, id_num]。
        window_size (tuple): 滑动窗口的大小 (宽度, 高度)，默认 (128, 128)。
        step_size (int): 滑动窗口的步幅，默认 127。

    返回:
        windows_images (list of ndarray): 切割后的窗口图像列表。
        windows_labels (list of lists): 对应于每个窗口的标签列表。
    """

    def read_labels(labels_file):
        with open(labels_file, 'r') as f:
            labels = [label.strip().split() for label in f.readlines()]  # 去掉每行空白字符并分割成列表
        return labels

    def sliding_window(image, step_size, window_size):
        """
        生成滑动窗口的生成器。

        参数:
            image (ndarray): 输入图像。
            step_size (int): 滑动步幅。
            window_size (tuple): 窗口大小 (宽度, 高度)。

        Yields:
            (x, y, window): 窗口左上角坐标 (x, y) 和窗口图像。
        """
        for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
            for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
                yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

    def bbox_in_window(bbox, window_x, window_y, window_w, window_h):
        """
        判断边界框的中心点是否在窗口内，并调整坐标。

        参数:
            bbox (list): 单个边界框 [x1, y1, x2, y2, x3, y3, x4, y4, class_name, id_num]。
            window_x (int): 窗口左上角 x 坐标。
            window_y (int): 窗口左上角 y 坐标。
            window_w (int): 窗口宽度。
            window_h (int): 窗口高度。
            id_num : 图片名
        Returns:
            list or None: 如果边界框中心在窗口内，返回调整后的边界框；否则返回 None。
        """
        class_name, x1, y1, x2, y2= bbox
        # x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
        # class_name, id_num = int(class_name), int(id_num)

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        if window_x <= center_x < window_x + window_w and window_y <= center_y < window_y + window_h:
            # 调整坐标相对于窗口
            new_coords = [
                x1 - window_x, y1 - window_y,
                x2 - window_x, y2 - window_y
            ]
            return [class_name] + new_coords
        return None

    image = cv2.imread(image_path)
    windows_images = []
    windows_labels = []

    for (x, y, window) in sliding_window(image, step_size, window_size):
        window_bboxes = []
        for bbox in labels:
            adjusted_bbox = bbox_in_window(bbox, x, y, window_size[0], window_size[1])
            if adjusted_bbox:
                window_bboxes.append(adjusted_bbox)

        if window_bboxes:
            windows_images.append(window)
            windows_labels.append(window_bboxes)

    return windows_images, windows_labels


# img_path = r"/coco/val2017/000000000139.jpg"
# labels_path = r"/coco/annfiles/000000000139.txt"
#
# win_img, win_label = sliding_window_segmentation_in_memory(img_path, labels_path)
# print('a')
