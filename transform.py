from data_pre.formatting import PackDetInputs
from data_pre.loading import LoadImageFromFile, LoadAnnotations
from data_pre.resize import YOLOv5KeepRatioResize, LetterResize

class Transform:
    def __init__(self):
        self.transforms = []
        self.transforms.append(LoadImageFromFile())
        self.transforms.append(YOLOv5KeepRatioResize(scale=(640, 640)))
        self.transforms.append(LetterResize(scale=(640, 640), allow_scale_up=False, pad_val={'img': 114}))
        self.transforms.append(LoadAnnotations(with_bbox=True))
        self.transforms.append(PackDetInputs(meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_param')))
    def __call__(self, img):
        for t in self.transforms:
            results = t(img)
        return results

