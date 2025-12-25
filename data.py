import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch


class COCOSegmentationDataset(Dataset):
    def __init__(self, coco, image_dir, transform=None):
        self.coco = coco
        self.image_dir = image_dir
        self.image_ids = coco.getImgIds()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])

        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann))

        # 同时接受image、mask
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # 如果图像仍然是numpy数组，则将其转换为张量CHW浮点数[0,1]
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)

        # 将掩码转换为形状为（1，H， W）且浮点值为0/1的张量
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        elif isinstance(mask, torch.Tensor):
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).float()
            elif mask.dim() == 3:
                # possible shapes: (H, W, 1) or (1, H, W) or (C, H, W)
                if mask.shape[0] == 1:
                    mask = mask.float()
                elif mask.shape[2] == 1:
                    mask = mask.permute(2, 0, 1).float()
                else:
                    # 如果有多个通道，取第一个通道
                    mask = mask[0:1].float()
        # 将掩码二值化
        mask = (mask > 0.5).float()

        return image, mask
