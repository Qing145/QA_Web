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
parser.add_argument('--output', type=str, default='preprocess')

args = parser.parse_args()
args.dete_model = os.path.join(args.output, args.dete_model)
args.entity_model = os.path.join(args.output, args.entity_model)
args.pred_model = os.path.join(args.output, args.pred_model)


def entity_predict(dataset_iter):
    model.eval()
    dataset_iter.init_epoch()
    gold_list = []
    pred_list = []
    dete_result = []
    question_list = []
    for data_batch_idx, data_batch in enumerate(dataset_iter):
        #batch_size = data_batch.text.size()[1]
        answer = torch.max(model(data_batch), 1)[1].view(data_batch.ed.size())
        # print('data_batch.ed.size(): ' + str(data_batch.ed.size()))
        # print(data_batch.ed.data)
        # print('data_batch.text.size(): ' + str(data_batch.text.size()))
        # print(data_batch.text.data)
        answer[(data_batch.text.data == 1)] = 1
        answer = np.transpose(answer.cpu().data.numpy())
        gold_list.append(np.transpose(data_batch.ed.cpu().data.numpy()))
        index_question = np.transpose(data_batch.text.cpu().data.numpy())
        question_array = index2word[index_question]
        dete_result.extend(answer)
        question_list.extend(question_array)
        pred_list.append(answer)
        break
    # P, R, F = evaluation(gold_list, pred_list, index2tag, type=False)
    # print("{} Precision: {:10.6f}% Recall: {:10.6f}% F1 Score: {:10.6f}%".format("Dev", 100. * P, 100. * R, 100. * F))
    return dete_result, question_list


def get_entity(dataset_iter):
    model.eval()
    dataset_iter.init_epoch()
    dete_result = []
    for data_batch_idx, data_batch in enumerate(dataset_iter):
        # batch_size = data_batch.text.size()[1]
        # print('data_batch.text.size(): ' + str(data_batch.text.size()))
        # print(data_batch.text.data)
        answer = torch.max(model(data_batch), 1)[1].view(data_batch.text.size())
        answer[(data_batch.text.data == 1)] = 1
        answer = np.transpose(answer.cpu().data.numpy())
        dete_result.extend(answer)
        break
    return dete_result


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


######################## Entity Detection  ########################
TEXT = data.Field(lower=True)
ED = data.Field()
#  75710 ############################################################  把 polyu 问题手动cv到 dete_train.txt
train = data.TabularDataset(path=os.path.join('freebase/', 'dete_train.txt'), format='tsv', fields=[('text', TEXT), ('ed', ED)])
# 设置为 none 则丢弃
field = [('id', None), ('sub', None), ('entity', None), ('relation', None), ('obj', None), ('text', TEXT), ('ed', ED)]
dev, test = data.TabularDataset.splits(path='freebase/', validation='valid.txt', test='test.txt', format='tsv', fields=field)
polyu_dataset = data.TabularDataset(path=os.path.join('preprocess/', 'formatted_question.txt'), format='tsv', fields=[('entity', None), ('relation', None), ('obj', None), ('text', TEXT), ('ed', None)])
polyu_dataset_dev = data.TabularDataset(path=os.path.join('preprocess/', 'whole_tabbed_question_only.txt'), format='tsv', fields=[('text', TEXT), ('ed', ED)])
TEXT.build_vocab(train, dev, test, polyu_dataset, polyu_dataset_dev)  # 构建词表时要加入polyu相关词 ######################################
ED.build_vocab(train, dev, polyu_dataset_dev)
total_num = len(test)
print('total num of example: {}'.format(total_num))

# load the model
if args.gpu == -1:  # Load all tensors onto the CPU
    # test_iter = data.Iterator(test, batch_size=args.batch_size, train=False, repeat=False, sort=False, shuffle=False,
    #                           sort_within_batch=False)
    model = torch.load(args.dete_model, map_location=lambda storage, loc: storage)
    model.config.cuda = False
else:
    # test_iter = data.Iterator(test, batch_size=args.batch_size, device=torch.device('cuda', args.gpu), train=False,
    #                           repeat=False, sort=False, shuffle=False, sort_within_batch=False)
    model = torch.load(args.dete_model, map_location=lambda storage, loc: storage.cuda(args.gpu))
index2tag = np.array(ED.vocab.itos)
# print(index2tag)
idxO = int(np.where(index2tag == 'O')[0][0])  # Index for 'O'  训练集dete_train.txt中的标记
idxI = int(np.where(index2tag == 'I')[0][0])  # Index for 'I'
index2word = np.array(TEXT.vocab.itos)

# run the model on the test set and write the output to a file
# dete_result, question_list = entity_predict(dataset_iter=test_iter)
#
#
# for i, question in enumerate(question_list):
#     if i < 2:  # 前两个句子不好
#         continue
#     question = [token for token in question if token != '<pad>']
#     print(question)
#     pred_span = get_span(dete_result[i], index2tag, type=False)
#     print(pred_span)
#     for i in pred_span:
#         start, end = i
#         print(question[start:end])
#
#     break

while True:
################################################################
    input_question = input("plz input your question: ")  # who recorded the song shake ya body ?
    input_question_list = input_question.split(' ')
    outfile = open('try.txt', "w", encoding='utf-8')
    outfile.write(input_question)
    outfile.close()
    # input_question = data.Dataset(examples=[input_question], fields=[('text', TEXT)])
    input_question = data.TabularDataset(path='try.txt', format='tsv', fields=[('text', TEXT)])
    input_question_iter = data.Iterator(input_question, batch_size=1, device=torch.device('cuda', args.gpu), train=False,
                                  repeat=False, sort=False, shuffle=False, sort_within_batch=False)

    # print(get_entity(input_question))

    dete_result = get_entity(dataset_iter=input_question_iter)
    pred_span = get_span(dete_result[0], index2tag, type=False)
    # pred_span = get_span(dete_result[0], ['<unk>' '<pad>' 'O' 'I'], type=False)
    print(pred_span)
    for i in pred_span:
        start, end = i
        print(input_question_list[start:end])



    # fileinput.input 有一个 inplace 参数，表示是否将标准输出的结果写回文件，默认不取代
    for line in fileinput.input('try.txt', inplace=True):
        if not fileinput.isfirstline():
            print(line.replace('\n', ''))


del model
