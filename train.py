import argparse
from typing import Dict
import time
import gc

from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD, Optimizer

from food_101_dataset import Food101Dataset, make_datapath_list
from vgg import vgg_a, get_vgg_model
from utils import seed_everything


SEED = 42
MIN_LR = 1e-5
batch_size = 128
num_workers = 0
num_epochs = 74


def valid(model: nn.Module, criterion: Optimizer, dataset: Dataset, 
         dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    
    test_loss = 0.0
    test_top1_corrects = 0

    for imgs, labels in tqdm(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        imgs = dataset.transform(imgs)

        with torch.no_grad():
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1) # top-1
        test_loss += loss.item() * imgs.size(0)
        test_top1_corrects += torch.sum(preds == labels.data)

        del imgs, labels, loss, outputs
        gc.collect()
        torch.cuda.empty_cache()

    test_loss = test_loss / len(dataloader.dataset)
    test_acc = test_top1_corrects.double() / len(dataloader.dataset)

    model.train()

    return {'loss': test_loss, 'acc': test_acc}


def main(args):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f'device is {device}')

    seed_everything(SEED)
    
    data_path = make_datapath_list()
    train_data_path, val_data_path= train_test_split(data_path, test_size=0.2, 
                                                     random_state=SEED)

    train_dataset = Food101Dataset(train_data_path, device=device)
    val_dataset = Food101Dataset(val_data_path, device=device)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, 
                          num_workers=num_workers, pin_memory=True, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, 
                          num_workers=num_workers, pin_memory=True, shuffle=False)
    
    model = get_vgg_model(args.model, len(train_dataset.label_list))
    model = model.to(device)

    if args.is_using_a:
        model_a = vgg_a(num_classes=len(train_dataset.label_list))
        model_a = model_a.to(device)
        model_a.load_state_dict(torch.load('./data/output/vgg_a_best.pth'))

        # 最初の4層のCNNと最後の3つの全結合層の重みをコピーする
        if args.model == 'vgg_a_lrn':
            model.state_dict()['features.0.weight'].copy_(model_a.state_dict()['features.0.weight'])
            model.state_dict()['features.4.weight'].copy_(model_a.state_dict()['features.3.weight'])
            model.state_dict()['features.7.weight'].copy_(model_a.state_dict()['features.6.weight'])
            model.state_dict()['features.9.weight'].copy_(model_a.state_dict()['features.8.weight'])
            model.state_dict()['features.0.bias'].copy_(model_a.state_dict()['features.0.bias'])
            model.state_dict()['features.4.bias'].copy_(model_a.state_dict()['features.3.bias'])
            model.state_dict()['features.7.bias'].copy_(model_a.state_dict()['features.6.bias'])
            model.state_dict()['features.9.bias'].copy_(model_a.state_dict()['features.8.bias'])
        else:
            # vgg_b, c, d, e
            model.state_dict()['features.0.weight'].copy_(model_a.state_dict()['features.0.weight'])
            model.state_dict()['features.5.weight'].copy_(model_a.state_dict()['features.3.weight'])
            model.state_dict()['features.10.weight'].copy_(model_a.state_dict()['features.6.weight'])
            model.state_dict()['features.12.weight'].copy_(model_a.state_dict()['features.8.weight'])
            model.state_dict()['features.0.bias'].copy_(model_a.state_dict()['features.0.bias'])
            model.state_dict()['features.5.bias'].copy_(model_a.state_dict()['features.3.bias'])
            model.state_dict()['features.10.bias'].copy_(model_a.state_dict()['features.6.bias'])
            model.state_dict()['features.12.bias'].copy_(model_a.state_dict()['features.8.bias'])

        del model_a
        gc.collect()
        torch.cuda.empty_cache()

    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    start_time = time.time()
    
    print(f'Start train {args.model}')
    for epoch in range(1, num_epochs+1):
        epoch_loss = 0.0
        epoch_top1_corrects = 0
        epoch_start_time = time.time()

        model.train()

        for imgs, labels in tqdm(train_dl):
            imgs, labels = imgs.to(device), labels.to(device)
            imgs = train_dataset.transform(imgs)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1) # top-1
            epoch_loss += loss.item() * imgs.size(0)
            epoch_top1_corrects += torch.sum(preds == labels.data)

            del imgs, labels, loss, outputs
            gc.collect()
            torch.cuda.empty_cache()
            #break

        val_result = valid(model, criterion, val_dataset, val_dl, device)

        if best_val_acc > val_result['acc']:
            # 学習率を低下させる
            if optimizer.param_groups[0]['lr'] > MIN_LR:
                optimizer.param_groups[0]['lr'] *= 0.1
                print(f'update optimizer lr {optimizer.param_groups[0]["lr"]}')
        else:
            best_val_acc = val_result['acc']
            torch.save(model.state_dict(), './data/output/vgg_e_best.pth')

        epoch_loss = epoch_loss / len(train_dl.dataset)
        epoch_acc = epoch_top1_corrects.double() / len(train_dl.dataset)

        print(f'{epoch}/{num_epochs} train Loss: {epoch_loss:.4f} train Acc: {epoch_acc:.4f}\
              Test Loss: {val_result["loss"]:.4f} Test Acc: {val_result["acc"]:.4f}\
              Epoch Time: {time.time()-epoch_start_time:.2f}')

    print(f'End train! Best Test Acc: {best_val_acc} Total Time: {time.time()-start_time:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VGG FOOD101 Test')
    parser.add_argument('--model', '-m', type=str, default='vgg_a', 
                        help='Please input "vgg_a" or "vgg_a_lrn" or "vgg_b" or\
                              "vgg_c" or "vgg_d" or "vgg_e"')
    parser.add_argument('--is_using_a', '-u', type=str, default=True, 
                        help='Do you use vgg_a weights in the first 4 CNN layers\
                              and the last 3 all-coupled layers"')
    args = parser.parse_args()

    main(args)