import mariadb 
from os import mkdir
from os.path import exists, join

from imblearn.over_sampling import RandomOverSampler, SMOTE
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import ssl
import torch
from torch.nn import BatchNorm1d, BCELoss, CrossEntropyLoss, Flatten, Identity, LazyBatchNorm1d, LazyLinear, Linear, Module, ReLU, Sequential, Dropout, Sigmoid
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.models import *
from torchvision import transforms

import sys
sys.path.append('..')
from data import SiameseThyroidDataset
from config import *


class CustomNet(Module):
    def __init__(self, backbone):
        super(CustomNet, self).__init__()
        if backbone in ['vgg16', 'vgg19']:
            self.backbone = eval(backbone)(pretrained=True)
        elif backbone in ['resnet18', 'resnet34']:
            self.backbone = eval(backbone)(pretrained=True)
        elif backbone in ['resnet50', 'resnet101']:
            self.backbone = eval(backbone)(pretrained=True)
        elif backbone in ['inception_v3']:
            self.backbone = eval(backbone)(pretrained=True, aux_logits=False)
        elif backbone in ['vit_b_16', 'vit_b_32']:
            self.backbone = eval(backbone)(pretrained=True)
        elif backbone in ['swin_t', 'swin_v2_t']:
            self.backbone = eval(backbone)(pretrained=True)

        self.backbone.fc = Identity()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.tail = Sequential(LazyLinear(500), LazyLinear(1), Sigmoid())

    def forward(self, x):
        x = self.backbone(x)
        x = self.tail(x)
        return x.squeeze()
    

def train_loop(dataloader, model, loss_fn, optimizer, scheduler, device):
    model.train()
    size = len(dataloader.dataset)
    total_train_loss = 0.0
    total_y = []
    total_pred = []
    for batch, (X, _, y, _, _)in enumerate(dataloader):
        total_y += y.tolist()
        X = X.to(device)
        y = y.float().to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        total_train_loss += loss.item()
        total_pred += pred.cpu().tolist()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    loss, current = loss.item(), batch * len(X)
    tn, fp, fn, tp = confusion_matrix(total_y, np.round(total_pred)).ravel()
    cls_rep = classification_report(total_y, np.round(total_pred), output_dict=True)
    acc = cls_rep['accuracy']
    precision = cls_rep['1']['precision']
    recall = cls_rep['1']['recall']
    f1 = cls_rep['1']['f1-score']
    print(f'Train Error:\nAvg loss: {total_train_loss:>7f}  [{current:>5d}/{size:>5d}]')
    print(f'TP: {tp}')
    print(f'TN: {tn}')
    print(f'FP: {fp}')
    print(f'FN: {fn}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Accuracy: {acc * 100:.2f}%')


def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[1] * (target * torch.log(output)) + weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))


if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    device = torch.device('cuda', int(sys.argv[1]))
    percent = int(sys.argv[2])
    
    transform = transforms.Compose([
        # transforms.Grayscale(3),
        transforms.Resize((image_size, ) * 2),
        transforms.ToTensor()
    ])

    data_dir = '../data/thyroid-images-croped-3'
    label_table_path = '../data/data-ti-c-3.csv'

    for model_name in ['vit_b_16', 'vit_b_32']:
        image_size = 224 if model_name != 'inception_v3' else 299
        if model_name in ['resnet18', 'resnet34']:
            batch_size = 4000
        elif model_name in ['resnet50']:
            batch_size = 3000
        elif model_name in ['resnet101']:
            batch_size = 800
        elif model_name in ['vgg16', 'vgg19', 'inception_v3']:
            batch_size = 300
        elif model_name in ['swin_t']:
            batch_size = 1000
        elif model_name in ['swin_v2_t']:
            batch_size = 3000
        elif model_name in ['vit_b_16']:
            batch_size = 3000
        elif model_name in ['vit_b_32']:
            batch_size = 2000
        else:
            batch_size = 8000
        for fold in range(slow_cv_folds):
            model = CustomNet(model_name).to(device)
            loss_fn = BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            scheduler = ExponentialLR(optimizer, 0.999)

            if not exists(weights_dir := f'weights-1227-{percent}'):
                mkdir(weights_dir)
            
            if not exists(wights_filename := join(weights_dir, f'{model_name}-e2e-{fold}.pth')):
                slow_train_ds = SiameseThyroidDataset(data_dir, label_table_path, 'slow_train', fast_set_size=fast_set_size, fast_slow_split_random_state=fast_slow_split_random_state, slow_valid_random_state=slow_valid_random_state, slow_cv_folds=slow_cv_folds, current_fold=fold, transform=transform, shrink_sample_random_state=0, shrink_keep_ratio=percent / 100)
                slow_train_dataloader = DataLoader(slow_train_ds, batch_size=batch_size, pin_memory=True, num_workers=16)

                epochs = 30
                for t in range(epochs):
                    print(f"Epoch {t + 1}\n-------------------------------")
                    train_loop(slow_train_dataloader, model, loss_fn, optimizer, scheduler, device)

                torch.save(model.state_dict(), wights_filename)
            else:
                model.load_state_dict(torch.load(wights_filename, map_location=device))
            
            slow_valid_ds = SiameseThyroidDataset(data_dir, label_table_path, 'slow_valid', fast_set_size=fast_set_size, fast_slow_split_random_state=fast_slow_split_random_state, slow_valid_random_state=slow_valid_random_state, slow_cv_folds=slow_cv_folds, current_fold=fold, transform=transform)
            slow_valid_dataloader = DataLoader(slow_valid_ds, batch_size=batch_size, pin_memory=True, num_workers=1)
            
            with mariadb.connect(**database_connection_params) as db:
                with db.cursor() as cursor:
                    total_y = []
                    total_pred = []
                    total_image = []
                    total_case = []
                    model.eval()
                    with torch.no_grad():
                        for X, _, y, image, case in slow_valid_dataloader:
                            total_y += y.tolist()
                            X = X.to(device)
                            y = y.to(device)
                            pred = model(X)
                            total_pred += pred.cpu().tolist()
                            total_image += image
                            total_case += case
                    for i, y in enumerate(total_y):
                        cursor.execute(f'INSERT INTO `deep-prediction` SET `model`=\'{model_name}\', `percent`={percent}, `case`=\'{total_case[i]}\', `image`=\'{total_image[i]}\', `true`={y}, `pred`={total_pred[i]};')
                    tn, fp, fn, tp = confusion_matrix(total_y, np.round(total_pred)).ravel()
                    cls_rep = classification_report(total_y, np.round(total_pred), output_dict=True)
                    acc = cls_rep['accuracy']
                    precision = cls_rep['1']['precision']
                    recall = cls_rep['1']['recall']
                    f1 = cls_rep['1']['f1-score']
                    specificity =  float(tn) / float(tn + fp)
                    auc = roc_auc_score(total_y, total_pred)
                    fprs, tprs, thresholds = roc_curve(total_y, total_pred, pos_label=1)
                    cursor.execute(f'INSERT INTO `deep-pred` SET `model`=\'{model_name}\', `fold`={fold}, `tp`={tp}, `tn`={tn}, `fp`={fp}, `fn`={fn}, `accuracy`={acc}, `recall`={recall}, `precision`={precision}, `specificity`={specificity}, `f1`={f1}, `percent`={percent}, `auc`={auc}, `fpr`=\'{",".join([str(fpr) for fpr in fprs])}\', `tpr`=\'{",".join([str(tpr) for tpr in tprs])}\', `thresholds`=\'{",".join([str(threshold) for threshold in thresholds])}\';')
                    db.commit()
