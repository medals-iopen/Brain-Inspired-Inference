from os.path import join

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.utils.random import sample_without_replacement
from config import *


class SiameseThyroidDataset(Dataset):
    def __init__(self, img_dir, label_table, epoch_type, fast_set_size=0.6, fast_slow_split_random_state=0, fast_valid_random_state=0, fast_cv_folds=5, slow_valid_random_state=0, slow_cv_folds=5, current_fold=0, transform=None, oversampled=False, undersampled=False, **kwargs):
        self.img_dir = img_dir
        self.label_table = pd.read_csv(label_table, encoding='utf-8', dtype={'id': str}) if isinstance(label_table, str) else label_table
        self.label_table.drop_duplicates(inplace=True)
        self.transform = transform

        if fast_set_size < 1.0:
            gss = GroupShuffleSplit(n_splits=1, train_size=fast_set_size, random_state=fast_slow_split_random_state)
            for fast_idx, slow_idx in gss.split(self.label_table[ti_rads], self.label_table['y'], self.label_table['case']):
                fast_set = self.label_table.iloc[fast_idx, :]
                slow_set = self.label_table.iloc[slow_idx, :]
        else:
            fast_set = self.label_table
            slow_set = None

        if 'target_class' in kwargs:
            self.target_class = kwargs['target_class']
        else:
            self.target_class = None
            
        if epoch_type.startswith('fast'):
            cv = StratifiedGroupKFold(n_splits=fast_cv_folds, random_state=fast_valid_random_state, shuffle=True)
            self.type = 'fast'
            train = fast_set
        elif epoch_type.startswith('slow'):
            self.type = 'slow'
            cv = StratifiedGroupKFold(n_splits=slow_cv_folds, random_state=slow_valid_random_state, shuffle=True)
            train = slow_set
        else:
            raise Exception(f'Epoch type {epoch_type} error!')

        if epoch_type.endswith('all'):
            self.X = train.drop('y', axis=1)
            self.y = train['y']
        else:
            for i, (train_index, valid_index) in enumerate(cv.split(X := train.drop('y', axis=1), y := train['y'], train['case'])):
                if i != current_fold:
                    continue
                if epoch_type.endswith('train'):
                    self.X = X.iloc[train_index, :]
                    self.y = y.iloc[train_index]
                    if 'shrink_sample_random_state' in kwargs and 'shrink_keep_ratio' in kwargs:
                        self.shrink_sample_random_state = kwargs['shrink_sample_random_state']
                        self.shrink_keep_ratio = kwargs['shrink_keep_ratio']
                        self.X = self.X.iloc[sample_without_replacement(len(self.X), int(len(self.X) * self.shrink_keep_ratio), random_state=self.shrink_sample_random_state), :]
                        self.y = self.y.iloc[sample_without_replacement(len(self.y), int(len(self.y) * self.shrink_keep_ratio), random_state=self.shrink_sample_random_state)]
                elif epoch_type.endswith('valid'):
                    self.X = X.iloc[valid_index, :]
                    self.y = y.iloc[valid_index]
        
        if oversampled == True:
            table = self.X.join(self.y).reset_index()
            ros = RandomOverSampler(random_state=0)
            sampeld_X, sampled_y = ros.fit_resample(table.drop(['y'], axis=1), table['y'])
            sampled_table = sampeld_X.join(sampled_y)
            sampled_table.set_index('index', inplace=True)
            sampled_table.index.name = None
            self.X = sampled_table.drop(['y'], axis=1)
            self.y = sampled_table['y']
        
        if undersampled == True:
            table = self.X.join(self.y).reset_index()
            rus = RandomUnderSampler(random_state=0)
            sampeld_X, sampled_y = rus.fit_resample(table.drop('y', 1), table['y'])
            sampled_table = sampeld_X.join(sampled_y)
            sampled_table.set_index('index', inplace=True)
            sampled_table.index.name = None
            self.X = sampled_table.drop('y', 1)
            self.y = sampled_table['y']
        
        self.index = self.X.index

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        if self.target_class is not None:
            class_code = int(self.X.iloc[idx, self.X.columns.get_loc(self.target_class)])
        else:
            class_code = 0
        label = int(self.y.iloc[idx])
        img = Image.open(join(self.img_dir, f'{self.X.iloc[idx, self.X.columns.get_loc("id")]}.jpg'))
        if self.transform is not None:
            img = self.transform(img)
        
        return img, class_code, label, self.X.iloc[idx, self.X.columns.get_loc('id')], self.X.iloc[idx, self.X.columns.get_loc('case')]
    
    def get_index(self):
        return self.index

    def get_index_col(self):
        return self.label_table.loc[self.index, 'id']
    
    def get_class_weight(self):
        origin_weight = 1.0 / self.X[self.target_class].value_counts().to_numpy()
        return origin_weight / origin_weight.sum()
    
    def get_final_tabel(self):
        return self.X.join(self.y, how='inner')