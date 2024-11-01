import torch
from tqdm import tqdm

from dataset import Ydataset
from models.model import Detector
from transform import Transform
from torch.utils.data import DataLoader
from dataloader import custom_collate_fn
from load import load_weights_with_mapping
import matplotlib.pyplot as plt


def plot_losses(cls_losses, bbox_losses, dff_losses, total_losses):
    epochs = range(1, len(cls_losses) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, cls_losses, 'r', label='Classification Loss')
    plt.plot(epochs, bbox_losses, 'b', label='Bounding Box Loss')
    plt.plot(epochs, dff_losses, 'g', label='Distribution Focal Loss')
    plt.plot(epochs, total_losses, 'm', label='Total Loss')

    plt.title('Training Losses Over Batches')
    plt.xlabel('Batches (x100)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def train_model(datasets, num_epochs, batch_size, device, ckp_dir, save_dir):
    yolo_model = Detector(batch_size=batch_size, optim="Adamw").to(device)
    yolo_model = load_weights_with_mapping(yolo_model, ckp_dir)
    dataloader = DataLoader(dataset=datasets, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    cls_losses = []
    bbox_losses = []
    dff_losses = []
    total_losses = []
    for epoch in range(num_epochs):
        yolo_model.train(True)
        pbar = tqdm(total=len(dataloader))
        cls_loss = 0
        bbox_loss = 0
        dff_loss = 0
        total_loss = 0
        for idx, batch in enumerate(dataloader):
            img = batch['input']
            datasample = batch['datasample']

            loss = yolo_model.step_train(img, datasample)
            cls_loss += loss['loss_cls']
            bbox_loss += loss['loss_bbox']
            dff_loss += loss['loss_dfl']
            total_loss += loss['total_loss']
            if (idx + 1) % 200 == 0 :
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{idx + 1}]")
                print(f"loss cls: {cls_loss / 100}")
                print(f"loss bbox: {bbox_loss / 100}")
                print(f"loss dfl: {dff_loss / 100}")
                avg_total_loss = total_loss / 100
                print(f"total_loss: {avg_total_loss}")
                cls_losses.append(cls_loss / 100)
                bbox_losses.append(bbox_loss / 100)
                dff_losses.append(dff_loss / 100)
                total_losses.append(avg_total_loss)
                # 保存模型
                filename = f'model_loss_{avg_total_loss:.4f}.pth'
                if epoch >= 3:
                    yolo_model.save_model(f'{save_dir}/{filename}')
                cls_loss = 0
                bbox_loss = 0
                dff_loss = 0
                total_loss = 0
            pbar.update(1)
        pbar.close()
    plot_losses(cls_losses, bbox_losses, dff_losses, total_losses)



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_dir = r'F:\study\CODEs\xiaotiao\coco\val2017'
    label_dir = r'F:\study\CODEs\xiaotiao\coco\annfiles'
    ckp_dir = 'yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6.pth'
    save_dir = 'save_model'
    datasets = Ydataset(img_dir, label_dir, transform=True, device=device)
    train_model(datasets, num_epochs=12, batch_size=4, device=torch.device('cuda'),ckp_dir=ckp_dir, save_dir=save_dir)
