from os.path import join

from load_data import Data
import mariadb
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from model import *
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
import argparse

import sys
sys.path.append('..')
from config import *

    
class Experiment:

    def __init__(self, ent_vec_dim=200, rel_vec_dim=200, num_iterations=500, batch_size=128, decay_rate=0., cuda=False, input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5, label_smoothing=0., device_id=0):
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.kwargs = {'input_dropout': input_dropout, 'hidden_dropout1': hidden_dropout1, 'hidden_dropout2': hidden_dropout2}
        self.device = torch.device('cuda', device_id)
        
    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs
    
    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets

    
    def bootstrap_auc(self, y, pred, bootstraps=100, fold_size=1000):
        statistics = np.zeros(bootstraps)

        df = pd.DataFrame(columns=['y', 'pred'])
        df.loc[:, 'y'] = y
        df.loc[:, 'pred'] = pred
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n = int(fold_size * (1 - prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            score = roc_auc_score(y_sample, pred_sample)
            statistics[i] = score
        return statistics


    def evaluate(self, n, slow_fold, ratio):
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}

        with mariadb.connect(**database_connection_params) as db:
            with db.cursor() as cursor:
                cursor.execute(f'SELECT `lr`, `epoch` FROM (SELECT * FROM `slow-valid` ORDER BY `f1` DESC LIMIT 99999999) `t` WHERE `t`.`ratio`={ratio} AND `t`.`fold`={slow_fold} GROUP BY `t`.`ratio`, `t`.`fold`;')
                best_lr, best_epoch = cursor.fetchone()
        model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        model.load_state_dict(torch.load(join(f'weights-{ratio}', f'{slow_fold},{slow_cv_folds}-{float(best_lr)}-{best_epoch}.pth'), map_location=self.device))
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            print("Validation:")
            test_data_idxs = self.get_data_idxs(d.valid_data)
            er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

            print(f'Number of data points: {len(test_data_idxs)}')
            
            total_y = []
            total_pred = []
            total_pred_prob = []
            total_index = []
            for i in range(0, len(test_data_idxs), self.batch_size):
                data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
                e1_idx = torch.tensor(data_batch[:,0])
                r_idx = torch.tensor(data_batch[:,1])
                e2_idx = torch.tensor(data_batch[:,2])
                
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                    e2_idx = e2_idx.cuda()
                predictions = model.forward(e1_idx, r_idx)
                batch_y = e2_idx.cpu().tolist()
                batch_pred_prob = predictions.cpu().tolist()
                for each_e1_idx in e1_idx:
                    for each_entity, each_entity_idx in self.entity_idxs.items():
                        if each_entity_idx == each_e1_idx:
                            total_index.append(' '.join(each_entity.split('_')))
                            break
                negative_idx = self.entity_idxs['y_0']
                positive_idx = self.entity_idxs['y_1']
                batch_pred_prob = softmax(np.array(batch_pred_prob)[:, [negative_idx, positive_idx]], 1)[:, 1]
                batch_y_bin = [0 if y == negative_idx else 1 for y in batch_y]
                total_y += batch_y_bin
                total_pred_prob += list(batch_pred_prob)
            with mariadb.connect(**database_connection_params) as db:
                with db.cursor() as cursor:
                    for i, y in enumerate(total_y):
                        cursor.execute(f'INSERT INTO `slow-prediction` SET `ratio`={ratio}, `case`=\'{total_index[i]}\', `true`={y}, `pred`={total_pred_prob[i]};')
                    db.commit()


if __name__ == '__main__':
    torch.set_num_threads(12)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237", nargs="?",
                    help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--num_iterations", type=int, default=1000, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=3000, nargs="?",
                    help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.01, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?",
                    help="Decay rate.")
    parser.add_argument("--edim", type=int, default=100, nargs="?",
                    help="Entity embedding dimensionality.")
    parser.add_argument("--rdim", type=int, default=100, nargs="?",
                    help="Relation embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--input_dropout", type=float, default=0, nargs="?",
                    help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0, nargs="?",
                    help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.1, nargs="?",
                    help="Dropout after the second hidden layer.")
    parser.add_argument("--label_smoothing", type=float, default=0, nargs="?",
                    help="Amount of label smoothing.")
    parser.add_argument("--start", type=float, default=0.01, nargs="?", help="lrs.")
    parser.add_argument("--end", type=float, default=0.021, nargs="?", help="lrs.")
    parser.add_argument("--device", type=int, default=0, nargs="?", help="CUDA device.")

    args = parser.parse_args()
    dataset = args.dataset
    
    for ratio in [100]:
        data_dir = f'data-{ratio}'
        torch.backends.cudnn.deterministic = True 
        seed = 0
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available:
            torch.cuda.manual_seed_all(seed)
        
        n = 1

        for i in range(1, n + 1):
            for slow_fold in range(slow_cv_folds):
                d = Data(data_dir=data_dir, reverse=False, fold=slow_fold)
                experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, 
                                        decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
                                        input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1, 
                                        hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing, device_id=args.device)
                experiment.evaluate(n=i, slow_fold=slow_fold, ratio=ratio)