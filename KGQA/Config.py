# JiangHao
class Config():
    def __init__(self):
        self.entity_detection_mode = 'LSTM'
        self.no_cuda = False
        self.gpu = -1
        self.seed = 3435
        self.dev_every = 12000
        self.log_every = 2000
        self.patience = 15
        self.dete_prefix = 'dete'
        self.words_dim = 300
        self.num_layer = 2
        self.rnn_fc_dropout = 0.3
        self.hidden_size = 300
        self.rnn_dropout = 0.3
        self.clip_gradient = 0.6
        self.vector_cache = "data/sq_glove300d.pt"
        self.weight_decay = 0
        self.fix_embed = False
        self.output = 'preprocess/'





