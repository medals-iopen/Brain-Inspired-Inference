from os.path import join

class Data:

    def __init__(self, data_dir='data', reverse=False, fold=0):
        self.train_data = self.load_data(data_dir, 'train', reverse=reverse, fold=fold)
        self.valid_data = self.load_data(data_dir, 'valid', reverse=reverse, fold=fold)
        self.test_data = self.load_data(data_dir, 'valid', reverse=reverse, fold=fold)
        self.data = self.train_data + self.valid_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.valid_data)
        self.relations = self.train_relations + [i for i in self.valid_relations if i not in self.train_relations]

    def load_data(self, data_dir, data_type='train', reverse=False, fold=0):
        with open(join(data_dir, f'best-1-{data_type}-({fold},5).txt'), 'r') as f:
            data_list = f.read().strip().split("\n")
            data = []
            for d in data_list:
                l = d.split()
                data.append(['_'.join(l[:-2]), l[-2], l[-1]])
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities