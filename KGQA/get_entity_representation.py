import torch
import numpy as np
import random
import os
import fileinput

from nltk.corpus import stopwords
from itertools import compress
from evaluation import evaluation, get_span
from argparse import ArgumentParser
# from torchtext import data
from torchtext.legacy import data
from sklearn.metrics.pairwise import euclidean_distances
from fuzzywuzzy import fuzz
from util import www2fb, processed_text, clean_uri

parser = ArgumentParser(description="Joint Prediction")
parser.add_argument('--no_cuda', action='store_false', help='do not use cuda', dest='cuda')
parser.add_argument('--gpu', type=int, default=0)  # Use -1 for CPU
parser.add_argument('--embed_dim', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seed', type=int, default=3435)
parser.add_argument('--dete_model', type=str, default='dete_best_model.pt')
parser.add_argument('--entity_model', type=str, default='entity_best_model.pt')
parser.add_argument('--pred_model', type=str, default='pred_best_model.pt')
# parser.add_argument('--output', type=str, default='preprocess')
parser.add_argument('--output', type=str, default='preprocess2')
args = parser.parse_args()
args.dete_model = os.path.join(args.output, args.dete_model)
args.entity_model = os.path.join(args.output, args.entity_model)
args.pred_model = os.path.join(args.output, args.pred_model)

# Set random seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for testing")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("Warning: You have Cuda but not use it. You are using CPU for testing.")



######################## Learn entity representation  ########################
TEXT = data.Field(lower=True)
ED = data.Field(sequential=False, use_vocab=False)
train, dev = data.TabularDataset.splits(path=args.output, train='entity_train.txt', validation='entity_valid.txt', format='tsv', fields=[('text', TEXT), ('mid', ED)])
field = [('id', None), ('sub', None), ('entity', None), ('relation', None), ('obj', None), ('text', TEXT), ('ed', None)]
test = data.TabularDataset(path=os.path.join(args.output, 'test.txt'), format='tsv', fields=field)
TEXT.build_vocab(train, dev, test)  # training data includes validation data

# load the model
model = torch.load(args.entity_model, map_location=lambda storage, loc: storage.cuda(args.gpu))

input_question = input("plz input your question: ")  # who recorded the song shake ya body ?
input_question_list = input_question.split(' ')
outfile = open('try.txt', "w", encoding='utf-8')
outfile.write(input_question)
outfile.close()

# input_question = data.Dataset(examples=[input_question], fields=[('text', TEXT)])
input_question = data.TabularDataset(path='try.txt', format='tsv', fields=[('text', TEXT)])
input_question_iter = data.Iterator(input_question, batch_size=1, device=torch.device('cuda', args.gpu), train=False,
                              repeat=False, sort=False, shuffle=False, sort_within_batch=False)

model.eval()
input_question_iter.init_epoch()

for data_batch_idx, data_batch in enumerate(input_question_iter):
    scores = model(data_batch).cpu().data.numpy()
    # print(scores)
    print(scores[0])
    print(len(scores[0]))
    break

del model


# fileinput.input 有一个 inplace 参数，表示是否将标准输出的结果写回文件，默认不取代
for line in fileinput.input('try.txt', inplace=True):
    if not fileinput.isfirstline():
        print(line.replace('\n', ''))