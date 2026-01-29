import os
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class CocoDetection(Dataset):
    def __init__(self, img_folder, ann_file, transforms=None):
        self.img_folder = img_folder
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # filter out annotations with area 0 or iscrowd=1
        anns = [ann for ann in anns if ann['area'] > 0 and ann['iscrowd'] == 0]

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_folder, path)).convert('RGB')

        # prepare target in coco api format
        w, h = img.size
        target = {
            'image_id': img_id,
            'annotations': anns,
            'size': torch.as_tensor([int(h), int(w)])
        }

        # apply transforms if any
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.ids)

def make_detr_transforms(image_set, image_size):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandomResize(scales, max_size=image_size),
            normalize,
        ])
    else: # val, test
        return T.Compose([
            T.RandomResize([image_size], max_size=image_size),
            normalize,
        ])

def collate_fn(batch):
    # custom collate_fn for DETR
    # images are padded to the largest image in the batch
    # targets are handled as a list
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets