import argparse
import time
import gc

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from food_101_dataset import Food101Dataset, make_datapath_list
from vgg import get_vgg_model
from utils import seed_everything


SEED = 42
MIN_LR = 1e-5
batch_size = 128
num_workers = 8
num_epochs = 74


def test(model, criterion, dataset, device):
    model.eval()
    test_dl = DataLoader(dataset, batch_size=batch_size, 
                          num_workers=num_workers, pin_memory=True, shuffle=False)
    
    test_loss = 0.0
    test_top1_corrects = 0
    test_top5_corrects = 0

    for imgs, labels in tqdm(test_dl):
        imgs, labels = imgs.to(device), labels.to(device)
        imgs = dataset.transform(imgs)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = F.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)

        test_loss += loss.item() * imgs.size(0)
        
        # 画像を左右反転させて同じことを行う
        imgs = torch.flip(imgs, dims=[3])

        with torch.no_grad():
            outputs_flip = model(imgs)
            outputs_flip = F.softmax(outputs_flip, dim=1)
            loss = criterion(outputs_flip, labels)

        test_loss += loss.item() * imgs.size(0)

        outputs = (outputs + outputs_flip) / 2
        _, preds_top1 = torch.max(outputs, 1) # top-1
        _, preds_top5 = torch.topk(outputs, k=5, dim=1)
        test_top1_corrects += torch.sum(preds_top1 == labels.data)
        test_top5_corrects += torch.sum((labels.view(-1, 1) == preds_top5).any(dim=1))

        del imgs, labels, loss, outputs
        gc.collect()
        torch.cuda.empty_cache()

    test_loss = test_loss / (len(test_dl.dataset)*2)
    test_acc_top1 = test_top1_corrects.double() / len(test_dl.dataset)
    test_acc_top5 = test_top5_corrects.double() / len(test_dl.dataset)

    model.train()

    return {'loss': test_loss, 'acc_top1': test_acc_top1, 'acc_top5': test_acc_top5}


def main(args):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f'device is {device}')

    seed_everything(SEED)
    
    test_data_path = make_datapath_list(is_train=False)
    test_dataset = Food101Dataset(test_data_path, device=device)
    
    model = get_vgg_model(args.model, len(test_dataset.label_list))
    model.load_state_dict(torch.load(f'./data/output/{args.model}_best.pth'))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    start_time = time.time()
 
    val_result = test(model, criterion, test_dataset, device)

    print(f'Test Loss: {val_result["loss"]:.4f} Test Acc Top1: {val_result["acc_top1"]:.4f}\
          Test Acc Top5: {val_result["acc_top5"]:.4f} Epoch Time: {time.time()-start_time:.2f}')
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VGG FOOD101 Test')
    parser.add_argument('--model', '-m', type=str, default='vgg_a', 
                        help='Please input "vgg_a" or "vgg_a_lrn" or "vgg_b" or\
                              "vgg_c" or "vgg_d" or "vgg_e"')
    args = parser.parse_args()

    main(args)