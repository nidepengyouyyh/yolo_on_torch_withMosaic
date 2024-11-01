import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_bboxes(image_tensor, labels, num_images=2):
    # 创建子图
    fig, axes = plt.subplots(1, num_images, figsize=(15, 7))

    for batch_idx in range(num_images):
        # 将张量转换为numpy数组并恢复为原始像素值范围（0-255）
        image = image_tensor[batch_idx].permute(1, 2, 0).cpu().numpy() * 255
        image = image.astype(np.uint8)
        image = image.copy()
        # 遍历每个标签，绘制边界框
        for bbox in labels[labels[:, 0] == batch_idx]:
            label = int(bbox[1].item())
            x1, y1, x2, y2 = bbox[2:].cpu().numpy()  # 边界框坐标
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色边框
            cv2.putText(image, str(label),(x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        # 在子图中显示图像
        axes[batch_idx].imshow(image)
        axes[batch_idx].axis('off')  # 不显示坐标轴

    plt.tight_layout()
    plt.show()