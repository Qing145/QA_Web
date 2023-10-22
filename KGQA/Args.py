# JiangHao
import os

class Args:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])

        self.cuda = False
        self.output = os.path.join(cur_dir, 'preprocess/')
        self.dete_model = 'dete_best_model.pt'
        self.entity_model = 'entity_best_model.pt'
        self.pred_model = 'pred_best_model.pt'
        self.gpu = -1
        self.embed_dim = 250
        self.batch_size = 16
        self.seed = 3435

        self.dete_model = os.path.join(self.output, self.dete_model)
        self.entity_model = os.path.join(self.output, self.entity_model)
        self.pred_model = os.path.join(self.output, self.pred_model)