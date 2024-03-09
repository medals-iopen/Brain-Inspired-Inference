import datetime
import sys
from collections import OrderedDict
from itertools import combinations
from os import mkdir
from os.path import exists
from sys import argv

import mariadb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

sys.path.append('..')
from config import *
from data import SiameseThyroidDataset
from utils import get_ip


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.kaiming_normal_(m.weight)
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        n = int((depth - 4) / 6)
        k = widen_factor
        filter = [16, 16 * k, 32 * k, 64 * k]
        self.tasks = len(num_classes)

        self.conv1 = conv3x3(3, filter[0], stride=1)
        self.layer1 = self._wide_layer(wide_basic, filter[1], n, stride=2)
        self.layer2 = self._wide_layer(wide_basic, filter[2], n, stride=2)
        self.layer3 = self._wide_layer(wide_basic, filter[3], n, stride=2)
        self.bn1 = nn.BatchNorm2d(filter[3], momentum=0.9)

        self.linear = nn.ModuleList([nn.Sequential(nn.Linear(2304, num_classes[0]), nn.Softmax(dim=1))])

        # attention modules
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])

        for j in range(self.tasks):
            if j < self.tasks - 1:
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))
                self.linear.append(nn.Sequential(nn.Linear(2304, num_classes[j + 1]), nn.Softmax(dim=1)))
            for i in range(3):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))

        for i in range(3):
            if i < 2:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))

    def conv_layer(self, channel):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=channel[1]),
            nn.ReLU(inplace=True),
        )
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return att_block

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, k):
        g_encoder = [0] * 4

        atten_encoder = [0] * self.tasks
        for i in range(self.tasks):
            atten_encoder[i] = [0] * 4
        for i in range(self.tasks):
            for j in range(4):
                atten_encoder[i][j] = [0] * 3

        # shared encoder
        g_encoder[0] = self.conv1(x)
        g_encoder[1] = self.layer1(g_encoder[0])
        g_encoder[2] = self.layer2(g_encoder[1])
        g_encoder[3] = F.relu(self.bn1(self.layer3(g_encoder[2])))

        # apply attention modules
        for j in range(4):
            if j == 0:
                atten_encoder[k][j][0] = self.encoder_att[k][j](g_encoder[0])
                atten_encoder[k][j][1] = (atten_encoder[k][j][0]) * g_encoder[0]
                atten_encoder[k][j][2] = self.encoder_block_att[j](atten_encoder[k][j][1])
                atten_encoder[k][j][2] = F.max_pool2d(atten_encoder[k][j][2], kernel_size=2, stride=2)
            else:
                atten_encoder[k][j][0] = self.encoder_att[k][j](torch.cat((g_encoder[j], atten_encoder[k][j - 1][2]), dim=1))
                atten_encoder[k][j][1] = (atten_encoder[k][j][0]) * g_encoder[j]
                atten_encoder[k][j][2] = self.encoder_block_att[j](atten_encoder[k][j][1])
                if j < 3:
                    atten_encoder[k][j][2] = F.max_pool2d(atten_encoder[k][j][2], kernel_size=2, stride=2)

        pred = F.avg_pool2d(atten_encoder[k][-1][-1], 8)
        pred = pred.view(pred.size(0), -1)

        out = self.linear[k](pred)
        return out

    def model_fit(self, x_pred, x_output, num_output, device):
        x_output_onehot = torch.zeros((len(x_output), num_output)).to(device)
        x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)

        loss = x_output_onehot * torch.log(x_pred + 1e-20)
        return torch.sum(-loss, dim=1)


def train_loop(classes, model, optimizer, avg_cost, epoch, device):
    model.train()
    for i, cls in enumerate(classes.keys()):
        cost = np.zeros(2, dtype=np.float32)
        size = len(train_cls_loader := classes[cls]['train_loader'])

        for batch, (X, y, _, _, _) in enumerate(train_cls_loader):
            X = X.to(device)
            y = y.to(device)
            pred = model(X, i)

            optimizer.zero_grad()
            loss = model.model_fit(pred, y, classes[cls]['size'], device)
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()

            train_predict_y = pred.data.max(1)[1]
            train_acc = train_predict_y.eq(y).sum().item() / X.shape[0]

            cost[0] = torch.mean(loss).item()
            cost[1] = train_acc
            avg_cost[epoch][i][:2] += cost / size

            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"Class {cls} loss: {loss:>7f}  [{current:>5d}/{size:>5d}] Train Accuracy: {train_acc * 100:.2f}")


def test_loop(classes, model, avg_cost, epoch, device):
    model.eval()
    with torch.no_grad():
        #test_accs = []
        for i, cls in enumerate(classes.keys()):
            cost = np.zeros(2, dtype=np.float32)
            size = len(test_cls_loader := classes[cls]['valid_loader'])
            total_y = []
            total_pred = []
            for batch, (X, y) in enumerate(test_cls_loader):
                total_y += y.tolist()
                X = X.to(device)
                y = y.to(device)
                pred = model(X, i)
                total_pred += pred.argmax(1).cpu().tolist()
                test_loss = model.model_fit(pred, y, classes[cls]['size'], device)
                test_loss = torch.mean(test_loss)
                test_predict_y = pred.data.max(1)[1]
                test_acc = test_predict_y.eq(y).sum().item() / X.shape[0]

                cost[0] = torch.mean(test_loss).item()
                cost[1] = test_acc
                avg_cost[epoch][i][2:] += cost / size

            cls_rep = classification_report(total_y, total_pred, output_dict=True)
            print(cls_rep)
            acc = cls_rep['accuracy']
            precision = cls_rep['weighted avg']['precision']
            recall = cls_rep['weighted avg']['recall']
            f1 = cls_rep['weighted avg']['f1-score']

            print(f'Valid Error:\nAvg loss: {test_loss:>8f}')
            print(f'Accuracy: {acc * 100:.2f}%, Precision: {precision * 100:.2f}, Recall: {recall * 100:.2f}, F1 Score: {f1 * 100:.2f}')


def train_process(label_table, database_table, classes, m, comb, comb_str, fold, folds, fast_set_size, fast_slow_split_random_state, transform, device_id, batch_size, epochs, weights_dir, database_connection_params, image_size=224):
    device = torch.device('cuda', device_id)

    for col in comb.keys():
        comb[col]['train_ds'] = SiameseThyroidDataset('../data/thyroid-images-croped-3', label_table, 'fast_train', fast_set_size=fast_set_size, fast_slow_split_random_state=fast_slow_split_random_state, target_class=col, current_fold=fold, transform=transform, oversampled=True)
        comb[col]['train_loader'] = DataLoader(comb[col]['train_ds'], batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
        comb[col]['valid_ds'] = SiameseThyroidDataset('../data/thyroid-images-croped-3', label_table, 'fast_valid', fast_set_size=fast_set_size, fast_slow_split_random_state=fast_slow_split_random_state, target_class=col, current_fold=fold, transform=transform)
        comb[col]['valid_loader'] = DataLoader(comb[col]['valid_ds'], batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)

    model = WideResNet(depth=16, widen_factor=4, num_classes=[cls['size'] for cls in comb.values()]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    avg_cost = np.zeros([20 + len(comb) * 10, 10, 4], dtype=np.float32)
    for t in range(20 + len(comb) * 10):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(comb, model, optimizer, avg_cost, t, device)
        with mariadb.connect(**database_connection_params) as db:
            with db.cursor() as cursor:
                cursor.execute(f'UPDATE `{database_table}` SET `epoch`={t + 1} WHERE `classes`=\'{comb_str}\' AND `fold`={fold};')
                db.commit()

    model.eval()
    with torch.no_grad():
        for i, col in enumerate(comb.keys()):
            total_y = []
            total_pred = []
            for batch, (X, y, _, _, _) in enumerate(valid_cls_loader := comb[col]['valid_loader']):
                total_y += y.tolist()
                X = X.to(device)
                y = y.to(device)
                pred = model(X, i)
                total_pred += pred.argmax(1).cpu().tolist()

            cls_rep = classification_report(total_y, total_pred, output_dict=True)
            acc = cls_rep['accuracy']
            precision = cls_rep['weighted avg']['precision']
            recall = cls_rep['weighted avg']['recall']
            f1 = cls_rep['weighted avg']['f1-score']
            with mariadb.connect(**database_connection_params) as db:
                with db.cursor() as cursor:
                    cursor.execute(f'INSERT INTO `{database_table}-valid` SET `class`=\'{col}\', `fold`={fold}, `classes`=\'{comb_str}\', `accuracy`={acc}, `recall`={recall}, `precision`={precision}, `f1`={f1};')
                    db.commit()


def train(device_id, batch_size, database_table, epochs=100, cv_folds=5, fast_set_size=0.6, fast_slow_split_random_state=0, image_size=224, stop_time=None, records_dir='records', weights_dir='weights-1205', **database_connection_params):
    label_table = pd.read_csv('../data/data.csv', encoding='utf-8', dtype={'id': str})
    print(label_table)

    transform = transforms.Compose([
        transforms.Resize((image_size, ) * 2),
        transforms.ToTensor()
    ])

    classes = OrderedDict()
    for col in ti_rads:
        classes[col] = {'size': len(label_table[col].drop_duplicates())}

    if not exists(weights_dir):
        mkdir(weights_dir)

    if stop_time:
        stop_time = datetime.datetime.strptime(stop_time, '%Y-%m-%d%H:%M')
    
    for m in range(1, 6):
        for comb in map(OrderedDict, combinations(classes.items(), m)):
            for fold in range(cv_folds):
                for target_class in comb:
                    if stop_time:
                        current_time = datetime.datetime.now()
                        if current_time > stop_time:
                            exit(0)
                    comb_str = ','.join(comb.keys())
                    with mariadb.connect(**database_connection_params) as db:
                        with db.cursor() as cursor:
                            cursor.execute(f'SELECT `epoch` FROM `{database_table}` WHERE `classes`=\'{comb_str}\' AND `fold`={fold};')
                            if cursor.fetchone():
                                continue
                            cursor.execute(f'INSERT INTO `{database_table}` SET `epoch`=0, `classes`=\'{comb_str}\', `fold`={fold}, `ip`=\'{get_ip()}\';')
                            db.commit()
                    train_process(label_table, database_table, classes, m, comb, comb_str, fold, cv_folds, fast_set_size, fast_slow_split_random_state, transform, int(device_id), int(batch_size), epochs, weights_dir, database_connection_params)


if __name__ == '__main__':
    # Args:
    # 1. DeviceID 
    # 2. Batch Size
    # 3. Stop Time

    database_table = 'fast'

    stop_time = argv[3] if len(argv) == 4 else None
    train(argv[1], argv[2], weights_dir='weights', database_table=database_table, epochs=fast_epochs, fast_set_size=fast_set_size, fast_slow_split_random_state=fast_slow_split_random_state, cv_folds=fast_cv_folds, stop_time=stop_time, **database_connection_params)