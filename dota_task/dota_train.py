import torch
from tqdm import tqdm

from dota_dataset import DOTA_dataset
from models.model import Detector
from torch.utils.data import DataLoader
from dataloader import custom_collate_fn
from load import load_weights_with_mapping, select_weights_with_mapping
def train_model(datasets, num_epochs, batch_size, device, ckp_dir, save_dir):
    yolo_model = Detector(batch_size=batch_size, num_class=15).to(device)
    yolo_model = select_weights_with_mapping(yolo_model, ckp_dir)
    dataloader = DataLoader(dataset=datasets, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
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
            if (idx + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{idx + 1}]")
                print(f"loss cls: {cls_loss / 100}")
                print(f"loss bbox: {bbox_loss / 100}")
                print(f"loss dfl: {dff_loss / 100}")
                cls_loss = 0
                bbox_loss = 0
                dff_loss = 0
                total_loss = 0
                yolo_model.save_model(save_dir)
            pbar.update(1)
        pbar.close()





if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_dir = '../data/DOTA/trainval/images-tiff'
    label_dir = '../data/DOTA/trainval/annfiles'
    ckp_dir = '../ckp/yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6.pth'
    save_dir = '../ckp_save/yolov8_dota.pth'
    datasets = DOTA_dataset(img_dir, label_dir, transform=True, device=device)
    train_model(datasets, num_epochs=20, batch_size=2, device=torch.device('cuda'),ckp_dir=ckp_dir, save_dir=save_dir)
