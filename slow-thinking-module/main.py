from os import mkdir
from os.path import exists, join

from load_data import Data
import mariadb
import numpy as np
import pandas as pd
import torch
import time
from collections import defaultdict
from model import *
from scipy.special import softmax
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from torch.optim.lr_scheduler import ExponentialLR
import argparse

import sys
sys.path.append('..')
from config import *

    
class Experiment:

    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200, num_iterations=500, batch_size=128, decay_rate=0., cuda=False, input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5, label_smoothing=0., device_id=0):
        self.learning_rate = learning_rate
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
    
    def evaluate(self, model, data, it, fold):
        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print(f'Number of data points: {len(test_data_idxs)}')
        
        total_y = []
        total_pred = []
        total_pred_prob = []
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
            total_y += e2_idx.cpu().tolist()
            total_pred_prob += predictions.cpu().tolist()
            

        negative_idx = self.entity_idxs['y_0']
        positive_idx = self.entity_idxs['y_1']

        total_pred_prob = softmax(np.array(total_pred_prob)[:, [negative_idx, positive_idx]], 1)
        total_pred = (total_pred_prob[:, 1] > 0.5) * 1
        total_y_bin = [0 if y == negative_idx else 1 for y in total_y]
        

        if it % 100 == 0:
            current_it = it
            with mariadb.connect(**database_connection_params) as db:
                with db.cursor() as cursor:
                    tn, fp, fn, tp = confusion_matrix(total_y_bin, total_pred).ravel()
                    cls_rep = classification_report(total_y_bin, total_pred, output_dict=True)
                    accuracy = cls_rep['accuracy']
                    precision = cls_rep['1']['precision']
                    recall = cls_rep['1']['recall']
                    f1 = cls_rep['1']['f1-score']
                    insert_sql = f'INSERT INTO `slow-valid` SET `epoch`={current_it}, `lr`={self.learning_rate}, `ratio`={ratio}, `fold`={fold}, `TP`={tp}, `FP`={fp}, `FN`={fn}, `TN`={tn}, `accuracy`={accuracy}'
                    if not np.isnan(recall):
                        insert_sql += f', `recall`={recall}'
                    if not np.isnan(precision):
                        insert_sql += f', `precision`={precision}'
                    if not np.isnan(f1):
                        insert_sql += f', `f1`={f1}'
                    specificity = np.float64(tn) / np.float64(tn + fp)
                    if not np.isnan(specificity):
                        insert_sql += f', `specificity`={specificity}'
                    try:
                        auc = roc_auc_score(total_y_bin, total_pred_prob[:, 1])
                        insert_sql += f', `auc`={auc}'
                        fprs, tprs, thresholds = roc_curve(total_y_bin, total_pred_prob[:, 1], pos_label=1)
                        insert_sql += f', `fpr`=\'{",".join([str(fpr) for fpr in fprs])}\', `tpr`=\'{",".join([str(tpr) for tpr in tprs])}\', `thresholds`=\'{",".join([str(threshold) for threshold in thresholds])}\''
                    except:
                        pass
                    insert_sql += ';'
                    cursor.execute(insert_sql)
                    db.commit()


    def train_and_eval(self, ratio, fold):
        print("Training the TuckER model...")
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        if self.cuda:
            model.to(self.device)
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        print("Starting training...")
        if not exists(weights_dir := f'weights-{ratio}'):
            mkdir(weights_dir)
            
        for it in range(1, self.num_iterations + 1):
            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:,0])
                r_idx = torch.tensor(data_batch[:,1])
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                predictions = model.forward(e1_idx, r_idx)
                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            print(f'Epoch {it}, loss: {np.mean(losses)}')
            
            if it % 100 == 0:
                model.eval()
                with torch.no_grad():
                    print("Validation:")
                    self.evaluate(model, d.valid_data, it, fold)
                    torch.save(model.state_dict(), join(weights_dir, f'{fold},5-{self.learning_rate}-{it}.pth'))


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
    parser.add_argument("--start", type=float, default=0.005, nargs="?", help="lrs.")
    parser.add_argument("--end", type=float, default=0.021, nargs="?", help="lrs.")
    parser.add_argument("--device", type=int, default=0, nargs="?", help="CUDA device.")

    args = parser.parse_args()
    dataset = args.dataset
    
    for ratio in reversed([100]):
        for lr in np.arange(args.start, args.end, 0.005):
            data_dir = f'data-{ratio}'
            torch.backends.cudnn.deterministic = True 
            seed = 0
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available:
                torch.cuda.manual_seed_all(seed)
            
            n = 1

            for fold in range(5):
                d = Data(data_dir=data_dir, reverse=False, fold=fold)
                experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=lr, 
                                        decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
                                        input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1, 
                                        hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing, device_id=args.device)
                experiment.train_and_eval(ratio=ratio, fold=fold)