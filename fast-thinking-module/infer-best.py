import sys
from collections import OrderedDict
from os import mkdir, remove
from os.path import exists, join
from shutil import rmtree
from sys import argv

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import AllKNN, RandomUnderSampler
import mariadb
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.utils.random import sample_without_replacement
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

sys.path.append('..')
from config import *
from data import SiameseThyroidDataset
from model import WideResNet


def valid_best(n, database_table, label_table, classes, image_size=224, infer_dir='infer', fast_set_size=0.5, **database_connection_params):
    valid_metric_list = []
    with mariadb.connect(**database_connection_params) as db:
        with db.cursor() as cursor:
            cursor.execute(f'SELECT `class`, `classes`, AVG(`accuracy`), AVG(`recall`), AVG(`precision`), AVG(`f1`) FROM `{database_table}-valid` GROUP BY `class`, `classes`;')
            for row in cursor.fetchall():
                cls, classes_str, accuracy, recall, precision, f1 = row
                # Limit task numbers
                if len(classes_str.split(',')) > 3:
                    continue
                valid_metric_list.append({'class': cls, 'classes': classes_str, 'accuracy': accuracy, 'recall': recall, 'precision': precision, 'f1': f1})
    valid_metric_table = pd.DataFrame(valid_metric_list)
    valid_metric_table.sort_values(['f1', 'accuracy', 'recall', 'precision'], ascending=False, inplace=True)
    
    best_models = {}
    for _, row in valid_metric_table.iterrows():
        if (cls := row['class']) not in best_models:
            best_models[cls] = []
        best_models[cls].append({'classes': row['classes'], 'f1': row['f1'], 'accuracy': row['accuracy'], 'recall': row['recall'], 'precision': row['precision']})
    
    with open(f'best-{n}.csv', 'w+', encoding='utf-8') as f:
        f.write('class;top_n;classes;f1;accuracy;recall;precision\n')
        for cls in best_models:
            for i in range(n):
                f.write(f'{cls};{i + 1};{best_models[cls][i]["classes"]};{best_models[cls][i]["f1"]};{best_models[cls][i]["accuracy"]};{best_models[cls][i]["recall"]};{best_models[cls][i]["precision"]}\n')


def train_loop(classes, model, optimizer, device):
    model.train()
    for i, cls in enumerate(classes.keys()):
        total_y = []
        total_pred = []
        for _, (X, y, _, _, _) in enumerate(classes[cls]['fast_train_loader']):
            total_y += y.tolist()
            X = X.to(device)
            y = y.to(device)
            pred = model(X, i)
            total_pred += pred.argmax(1).cpu().tolist()

            optimizer.zero_grad()
            loss = model.model_fit(pred, y, classes[cls]['size'], device)
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()

        cls_rep = classification_report(total_y, total_pred, output_dict=True)
        acc = cls_rep['accuracy']
        precision = cls_rep['weighted avg']['precision']
        recall = cls_rep['weighted avg']['recall']
        f1 = cls_rep['weighted avg']['f1-score']
        print(f'Acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}')


def infer_by_best(label_table, best_table_prefix, weights_dir, classes, infer_dir, ratio, device_id=0, oversampled=False, undersampled=False):
    device = torch.device('cuda', device_id)

    if exists(infer_dir):
        rmtree(infer_dir)
    mkdir(infer_dir)

    if not exists(weights_dir):
        mkdir(weights_dir)

    transform = transforms.Compose([
        transforms.Resize((image_size,) * 2),
        transforms.ToTensor()
    ])

    best_table = pd.read_csv(f'{best_table_prefix}-1.csv', encoding='utf-8', delimiter=';')
    print(best_table)

    gss = GroupShuffleSplit(n_splits=1, train_size=fast_set_size, random_state=fast_slow_split_random_state)
    for _, slow_idx in gss.split(label_table[ti_rads], label_table['y'], label_table['case']):
        slow_set = label_table.iloc[slow_idx, :]

    print(slow_set)

    for col in classes.keys():
        classes[col]['fast_train_ds'] = SiameseThyroidDataset('../data/thyroid-images-croped-3', label_table, 'fast_all', fast_set_size=fast_set_size, fast_cv_folds=fast_cv_folds, fast_slow_split_random_state=fast_slow_split_random_state, fast_valid_random_state=fast_valid_random_state, target_class=col, transform=transform, oversampled=True)
        classes[col]['fast_train_loader'] = DataLoader(classes[col]['fast_train_ds'], batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=12)
    
    cv = StratifiedGroupKFold(n_splits=slow_cv_folds, random_state=slow_valid_random_state, shuffle=True)

    for _, row in best_table.iterrows():
        current_classes = row['classes'].split(',')
        print(current_classes)
        model = WideResNet(depth=16, widen_factor=4, num_classes=[classes[col]['size'] for col in current_classes]).to(device)
        best_weight_placeholder = join(weights_dir, f'{row["classes"]}.pth.temp')
        if not exists(best_weight := join(weights_dir, f'{row["classes"]}.pth')) and not exists(best_weight_placeholder):
            torch.save(model.state_dict(), best_weight_placeholder) 
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            comb = {k: classes[k] for k in current_classes}
            for t in range(50):
                print(f"Epoch {t + 1}\n-------------------------------")
                train_loop(comb, model, optimizer, device)
            torch.save(model.state_dict(), best_weight)
            remove(best_weight_placeholder)
        else:
            if exists(best_weight_placeholder):
                continue
            model.load_state_dict(torch.load(best_weight, map_location=device))
            model.to(device)
        model.eval()
        with torch.no_grad():
            for i, (train_index, valid_index) in enumerate(cv.split(X := slow_set.drop('y', axis=1), y := slow_set['y'], slow_set['case'])):
                train_part = X.iloc[train_index, :].join(y.iloc[train_index])
                train_part = train_part.iloc[sample_without_replacement(len(train_part), int(len(train_part) * ratio), random_state=0), :]
                valid_part = X.iloc[valid_index, :].join(y.iloc[valid_index])
                if oversampled == True:
                    ros = RandomOverSampler(random_state=0)
                    sampeld_X, sampled_y = ros.fit_resample(train_part.drop(labels=['y'], axis=1), train_part['y'])
                    sampled_table = sampeld_X.join(sampled_y)
                    train_part_os_X = sampled_table.drop('y', 1)
                    train_part_os_y = sampled_table['y']
                    train_part = train_part_os_X.join(train_part_os_y)
                if undersampled == True:
                    rus = AllKNN('majority', 5)
                    sampeld_X, sampled_y = rus.fit_resample(train_part.drop(labels=['index', 'y'], axis=1), train_part['y'])
                    sampeld_X.index = train_part.drop(labels=['y'], axis=1).index[rus.sample_indices_]
                    sampled_y.index = train_part['y'].index[rus.sample_indices_]
                    sampled_table = sampeld_X.join(sampled_y)
                    train_part_os_X = sampled_table.drop(labels=['y'], axis=1)
                    train_part_os_y = sampled_table['y']
                    all_train_index = train_part['index']
                    train_part = train_part_os_X.join(train_part_os_y).join(all_train_index)
                total_table = pd.concat([train_part, valid_part]).reset_index(drop=True).set_index('id')
                total_table.index.name = None
                train_part = total_table.iloc[:len(train_part), :]
                valid_part = total_table.iloc[len(train_part):, :]
                train_part.to_csv(join(infer_dir, f'({i},{slow_cv_folds})-train.csv'), encoding='utf-8')
                
                slow_valid_pred_dict = {}
                for j, cls in enumerate(current_classes):
                    if cls != row['class']:
                        continue
                    total_slow_valid_pred = []
                    total_slow_valid_y = []
                    batch = []
                    c = 0
                    for valid_idx, valid_row in valid_part.iterrows():
                        img = Image.open(join('../data/thyroid-images-croped-3', f'{valid_idx}.jpg'))
                        batch.append(transform(img))
                        if len(batch) >= valid_batch_size or c == len(valid_part) - 1:
                            batch_tensor = torch.stack(batch).to(device)
                            batch.clear()
                            total_slow_valid_pred += model(batch_tensor, j).argmax(axis=1).cpu().tolist()
                        total_slow_valid_y.append(valid_row['y'])
                        c += 1
                    slow_valid_pred_dict[cls] = total_slow_valid_pred
                    slow_valid_pred_dict['y'] = total_slow_valid_y
                    slow_valid_pred_df = pd.DataFrame(slow_valid_pred_dict, index=valid_part.index)
                    slow_valid_pred_df.to_csv(join(infer_dir, f'{row["classes"]}-{cls}-({i},{slow_cv_folds})-valid.csv'), encoding='utf-8')
                    break

        
if __name__ == '__main__':
    device_id = int(argv[1])
    train_batch_size = int(argv[2])
    valid_batch_size = int(argv[3])

    database_table = 'fast-1227'

    label_table_path = '../data/data.csv'
    label_table = pd.read_csv(label_table_path, encoding='utf-8', dtype={'id': str})
    print(label_table)

    classes = OrderedDict()
    for col in ti_rads:
        classes[col] = {'size': len(label_table[col].drop_duplicates())}

    valid_best(1, database_table, label_table_path, classes, image_size=image_size, fast_set_size=fast_set_size, **database_connection_params)
    valid_best(3, database_table, label_table_path, classes, image_size=image_size, fast_set_size=fast_set_size, **database_connection_params)
    for r in range(int(argv[4]), int(argv[5]), 10):
        infer_by_best(label_table, 'best', 'weights', classes, f'infer-{r}', r / 100, device_id=device_id)
