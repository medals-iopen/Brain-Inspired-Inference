import sys
from collections import OrderedDict
from os import mkdir
from os.path import exists, join

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

sys.path.append('..')
from config import *


def concat_cumcount(original_value, c):
    if c > 0:
        return f'{original_value}_{c}'
    else:
        return original_value


if __name__ == '__main__':
    for ratio in [100]:
        infer_dir = f'infer-1227-{ratio}'
        n = 1
        target_dir = f'../slow-thinking-module/data-{ratio}'

        label_table_path = '../data/data.csv'
        label_table = pd.read_csv(label_table_path, encoding='utf-8', index_col=0)

        classes = OrderedDict()
        for col in ti_rads:
            classes[col] = {'size': len(label_table[col].drop_duplicates())}

        best_table = pd.read_csv(f'best-{n}.csv', encoding='utf-8', delimiter=';')

        for i in range(n):
            best_table_top_i = best_table.where(best_table['top_n'] == i + 1)
            for slow_fold in range(slow_cv_folds):
                # slow_train_dfs = []
                slow_valid_dfs = []
                for idx, row in best_table_top_i.dropna().iterrows():
                    current_classes = row['classes'].split(',')
                    slow_valid_dfs.append(pd.read_csv(join(infer_dir, f'{row["classes"]}-{row["class"]}-({slow_fold},{slow_cv_folds})-valid.csv'), encoding='utf-8', index_col=0))
                slow_train_df = pd.read_csv(join(infer_dir, f'({slow_fold},{slow_cv_folds})-train.csv'), index_col=0)

                # Oversampling for training set
                print(slow_train_df)
                slow_train_df.reset_index(inplace=True)
                slow_train_df_os_X, slow_train_df_os_y = RandomUnderSampler(random_state=0).fit_resample(slow_train_df.drop(labels=['y'], axis=1), slow_train_df['y'])
                slow_train_df = slow_train_df_os_X.join(slow_train_df_os_y)
                slow_train_df['case_cumcount'] = slow_train_df.groupby('case').cumcount()
                slow_train_df['case'] = slow_train_df.apply(lambda x: concat_cumcount(x['case'], x['case_cumcount']), axis=1)
                slow_train_df = slow_train_df.drop(labels=['case_cumcount'], axis=1)

                slow_valid_df = pd.concat(slow_valid_dfs, axis=1, join='inner')
                slow_valid_df_y = slow_valid_df['y'].iloc[:, 0]
                slow_valid_df.drop('y', axis=1, inplace=True)
                slow_valid_df['y'] = slow_valid_df_y

                print(slow_train_df)
                print(slow_valid_df)
                case_col = label_table['case'].copy()

                slow_train_df = slow_train_df.groupby('case').agg({'case': 'first', 'label': 'first', 'composition': 'first', 'echogenicity': 'first', 'shape': 'first', 'margin': 'first', 'echogenic foci': 'first', 'y': 'first'})
                slow_valid_df = pd.merge(case_col, slow_valid_df, left_index=True, right_index=True)
                slow_valid_df = slow_valid_df.groupby('case').agg({'case': 'first', 'composition': 'max', 'echogenicity': 'max', 'shape': 'max', 'margin': 'max', 'echogenic foci': 'max', 'y': 'first'})
                for t in ti_rads:
                    slow_valid_df[t] = slow_valid_df[t].astype(int)

                print(slow_train_df)
                print(slow_valid_df)

                if not exists(target_dir):
                    mkdir(target_dir)
                
                with open(join(target_dir, f'best-{i + 1}-train-({slow_fold},{slow_cv_folds}).txt'), 'w+', encoding='utf-8') as f:
                    for idx, row in slow_train_df.iterrows():
                        for col in classes:
                            f.write(f'{idx} {col.replace(" ", "_")} {col.replace(" ", "_")}_{row[col]}\n')
                        f.write(f'{idx} y y_{row["y"]}\n')
                    for idx, row in slow_valid_df.iterrows():
                        for col in classes:
                            f.write(f'{idx} {col.replace(" ", "_")} {col.replace(" ", "_")}_{row[col]}\n')
                
                with open(join(target_dir, f'best-{i + 1}-valid-({slow_fold},{slow_cv_folds}).txt'), 'w+', encoding='utf-8') as f:
                    for idx, value in slow_valid_df['y'].items():
                        f.write(f'{idx} y y_{value}\n')