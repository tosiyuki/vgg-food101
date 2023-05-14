from typing import List, Tuple
import os

from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms as transforms


class Food101Dataset(data.Dataset):
    def __init__(self, file_list: List[str], scale: int=256, 
                 is_train: bool=True, device=torch.device('cpu')) -> None:
        super().__init__()
        
        self.label_list = _load_metadata('classes.txt')
        self.scale = scale
        self.file_list = file_list

        if is_train:
            self.transform = nn.Sequential(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224),
            ).to(device)
        else:
            self.transform = nn.Sequential(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225]),
                transforms.RandomCrop(224),
            ).to(device)

        self._getter_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, str]:
        img_path = self.file_list[index]
        
        img = Image.open(img_path).convert('RGB').resize((self.scale, self.scale))
        img = self._getter_transform(img)

        label = os.path.split(os.path.dirname(img_path))[-1]
        
        return img, self.label_list.index(label)

def _load_metadata(metadata_path: str) -> List[str]:
    metadata_path = os.path.join('./data/food101/meta/meta', metadata_path)

    with open(metadata_path) as f:
        return [i.rstrip()for i in f.readlines()]
    
def make_datapath_list(is_train: bool=True) -> List[str]:
    if is_train:
        file_list = _load_metadata('train.txt')
    else:
        file_list = _load_metadata('test.txt')
    
    target_path = './data/food101/images'
    path_list = [os.path.join(target_path, path+'.jpg') for path in file_list]

    return path_list


if __name__ == '__main__':
    train_path_list = make_datapath_list()
    print(train_path_list[0])
    train_dataset = Food101Dataset(train_path_list)
    print(train_dataset.transform(train_dataset[0][0]))
    print(train_dataset[10000][1])