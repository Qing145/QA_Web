import torch
import numpy as np
import random
import os
import fileinput
import json
import string

import math
from collections import Counter
import nltk

from nltk.corpus import stopwords
from itertools import compress
from evaluation import evaluation, get_span
from argparse import ArgumentParser
# from torchtext import data
from torchtext.legacy import data
from sklearn.metrics.pairwise import euclidean_distances
from fuzzywuzzy import fuzz
from util import www2fb, processed_text, clean_uri

'''
相比 v8， 改进了 fuzzy 代码； 将简写表换成了别名表
原版本fuzzy: diy_fuzzy()
elk版本fuzzy: EntityLinking()
'''

parser = ArgumentParser(description="Joint Prediction")
parser.add_argument('--no_cuda', action='store_false', help='do not use cuda', dest='cuda')
# parser.add_argument('--no_cuda', action='store_true', help='do not use cuda', dest='cuda')
parser.add_argument('--gpu', type=int, default=0)  # Use -1 for CPU
# parser.add_argument('--gpu', type=int, default=-1)  # Use -1 for CPU
parser.add_argument('--embed_dim', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seed', type=int, default=3435)
parser.add_argument('--dete_model', type=str, default='dete_best_model.pt')
parser.add_argument('--entity_model', type=str, default='entity_best_model.pt')
parser.add_argument('--pred_model', type=str, default='pred_best_model.pt')
parser.add_argument('--output', type=str, default='preprocess')
# parser.add_argument('--output', type=str, default='preprocess2')
args = parser.parse_args()
args.dete_model = os.path.join(args.output, args.dete_model)
args.entity_model = os.path.join(args.output, args.entity_model)
args.pred_model = os.path.join(args.output, args.pred_model)


def get_entity(dataset_iter):
    model.eval()
    dataset_iter.init_epoch()
    dete_result = []
    for data_batch_idx, data_batch in enumerate(dataset_iter):
        # batch_size = data_batch.text.size()[1]
        answer = torch.max(model(data_batch), 1)[1].view(data_batch.text.size())
        answer[(data_batch.text.data == 1)] = 1
        answer = np.transpose(answer.cpu().data.numpy())
        dete_result.extend(answer)
        break
    return dete_result


def compute_reach_dic(matched_mid):  # 输入为 所有 matched 的 head_name
    reach_dic = {}
    with open(os.path.join('preprocess/', 'formatted_question.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            items = line.strip().split("\t")
            head_name = items[0].lower()  # 记得转换为小写
            if head_name in matched_mid and items[1].lower() in pre_dic:
                if reach_dic.get(head_name) is None:
                    reach_dic[head_name] = [pre_dic[items[1].lower()]]
                else:
                    reach_dic[head_name].append(pre_dic[items[1].lower()])
    return reach_dic  # 返回 reach_dic[head_name] = [pred idx 1, pred idx 2, ...]


class diy_fuzzy:
    def __init__(self):
        """
        加载所有的entity存入entities
        把所有entity, simplified word变成token存入word_list
        """
        self.entities = set()
        self.word_list = []
        self.stop_words = [" - ", "the", "a", "an"]

        with open("preprocess/train.txt", "r", encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                arr = line.split("|")
                if arr[0] not in self.entities:
                    tokens = nltk.word_tokenize(arr[0].lower().strip())
                    for word in self.stop_words:
                        if word in tokens:
                            tokens.remove(word)

                    if " " in tokens:
                        tokens.remove(" ")

                    self.entities.add(" ".join(tokens))
                    self.word_list.append(tokens)

        with open("preprocess/alias.txt", "r", encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                arr = line.strip("\n").split("|")
                # 有别名
                if arr[1] != "":
                    alias_list = arr[1].split(",")
                    # 将所有别名以键值对加进alias
                    for alias_item in alias_list:
                        # 把别名里的stop words都处理了
                        tokens = nltk.word_tokenize(alias_item.lower().strip())
                        for word in self.stop_words:
                            if word in tokens:
                                tokens.remove(word)
                        if " " in tokens:
                            tokens.remove(" ")

                        self.entities.add(" ".join(tokens))
                        self.word_list.append(tokens)


        # countlist：每个单词的tf的集合
        # word_list：所有entity包含单词的集合（去除重复的单词）
        self.countlist = []
        for i in range(len(self.word_list)):
            count = Counter(self.word_list[i])
            self.countlist.append(count)

        self.word_list = set(sum(self.word_list, []))

    def tf(self, word, count):
        return count[word] / sum(count.values())

    # 统计的是含有该单词的句子数
    def n_containing(self, word, count_list):
        return sum(1 for count in count_list if word in count)

    # len(count_list)是指句子的总数，n_containing(word, count_list)是指含有该单词的句子的总数，加1是为了防止分母为0
    def idf(self, word, count_list):
        if self.n_containing(word, count_list) == 0:
            return 0

        return math.log(len(count_list) / (1 + self.n_containing(word, count_list)))

    def computation(self, string1, string2):
        # score = fuzz.token_sort_ratio(string1, string2)  # 初稿版
        score = (fuzz.token_sort_ratio(string1, string2) + fuzz.token_set_ratio(string1, string2)) / 2
        return score

    def get_ent_from_fuzzy(self, input_word):
        target_word = input_word
        target_tokens = nltk.word_tokenize(target_word)
        for word in self.stop_words:
            if word in target_tokens:
                target_tokens.remove(word)
        if " " in target_tokens:
            target_tokens.remove(" ")
        target_word = " ".join(target_tokens)

        # 根据idf，保留top-n target tokens
        if len(target_tokens) > 1:
            target_token_top_n = 2
        else:
            target_token_top_n = 1
        target_dics = []
        for index, word in enumerate(target_tokens):
            dic = {
                "word": "",
                "idf_score": 0
            }
            idf_score = self.idf(word, self.countlist)
            dic["word"] = word
            dic["idf_score"] = idf_score
            target_dics.append(dic)

        # 找出 idf 最高的几个单词
        target_dics = sorted(target_dics, key=lambda k: k['idf_score'], reverse=True)

        # 保留top-n candidate words
        top_n = 2
        word_candidates = []
        for i in range(target_token_top_n):
            word_candidates.append([{"string": "", "score": 0}] * top_n)
        for index, target_token in enumerate(target_dics[:target_token_top_n]):
            for word in self.word_list:
                word_candidates[index] = sorted(word_candidates[index], key=lambda k: k['score'], reverse=True)
                score = fuzz.ratio(word, target_token['word'])
                if score > word_candidates[index][-1]["score"]:
                    word_candidates[index].pop()
                    word_candidates[index].append({"string": word, "score": score})

        # 二维list变一维
        word_candidates = [j for i in word_candidates for j in i]

        # 利用candidate words获取candidate entities
        candidate_entities = []
        for entity in self.entities:
            arr = nltk.word_tokenize(entity)
            for i in range(len(word_candidates)):
                if word_candidates[i]['string'] in arr:
                    candidate_entities.append(entity)
                    break
        candidate_entities = set(candidate_entities)

        """
        在candidate entities中利用fuzzy获得top-n entities，此处是利用整个句子下去匹配
        """
        top_n = 5
        scores = [{"string": "", "score": 0}] * top_n
        # target = nltk.word_tokenize(target_word)
        # length_target = len(target)

        for string1 in candidate_entities:
            scores = sorted(scores, key=lambda k: k['score'], reverse=True)
            score = self.computation(string1, target_word)
            if score > scores[len(scores) - 1]['score']:
                scores.pop()
                scores.append({"string": string1, "score": score})

        # # 只返回得分最高的
        # scores = sorted(scores, key=lambda k: k['score'], reverse=True)
        # final_result = scores[0]['string']
        # max_score = scores[0]['score']
        # if max_score >= 75:
        #     return final_result  # 可能得到的是缩写, 所以还要将缩写转换为一般entity
        # else:
        #     return None

        # 返回所有大于 threshold 的
        res = []
        for s in scores:
            if s['score'] >= 75:
                res.append(s['string'])
        return res


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


######################## Prepare some dictionary  ########################
# 构建 entity:entityidx  entityidx:entity  pred:predidx  predidx:pred  的字典  ###########################
mid_dic, mid_num_dic = {}, {}  # Dictionary for MID     entity:entityidx
for line in open(os.path.join('preprocess/', 'entity2id.txt'), 'r', encoding='utf-8'):
    items = line.strip().split("|")
    mid_dic[items[0].lower()] = int(items[1])
    mid_num_dic[int(items[1])] = items[0].lower()
pre_dic, pre_num_dic = {}, {}  # Dictionary for predicates   pred:predidx
match_pool = []  # 存 relation 实体词
for line in open(os.path.join('preprocess/', 'relation2id.txt'), 'r', encoding='utf-8'):
    items = line.strip().split("|")
    match_pool = match_pool + items[0].lower().split()
    pre_dic[items[0].lower()] = int(items[1])
    pre_num_dic[int(items[1])] = items[0].lower()
# 加入 stopwords
match_pool = set(match_pool + stopwords.words('english') + ["'s"])


# Embedding for MID
n = 0
for line in open(os.path.join('preprocess/', 'entity_250dim.txt'), 'r', encoding='utf-8'):
    items = line.strip().split("\t")
    if len(items) != 2:
        print(items)
        continue
    n += 1

arr = [[]for _ in range(n)]
for line in open(os.path.join('preprocess/', 'entity_250dim.txt'), 'r', encoding='utf-8'):
    items = line.strip().split("\t")
    if len(items) != 2:
        print(items)
        continue
    # 现在 items[1] 是一堆字符串, 需要处理
    embedding = json.loads(items[1])
    arr[int(items[0])] = embedding  # idx 对应原本 entity 的编号
    # arr.append(items[1])
entities_emb = np.array(arr, dtype=np.float32)

n = 0
for line in open(os.path.join('preprocess/', 'relation_250dim.txt'), 'r', encoding='utf-8'):
    items = line.strip().split("\t")
    if len(items) != 2:
        print(items)
        continue
    n += 1
arr = [[]for _ in range(n)]
for line in open(os.path.join('preprocess/', 'relation_250dim.txt'), 'r', encoding='utf-8'):
    items = line.strip().split("\t")
    if len(items) != 2:
        print(items)
        continue
    # 现在 items[1] 是一堆字符串, 需要处理
    embedding = json.loads(items[1])
    arr[int(items[0])] = embedding  # idx 对应原本 entity 的编号
    # arr.append(items[1])
predicates_emb = np.array(arr, dtype=np.float32)

# 构建 简写head:[head] 的dictionary  (存在一个别名对应多个实体的存在)
simplified_head = dict()
for line in open(os.path.join('preprocess/', 'alias.txt'), 'r', encoding='utf-8'):
    items = line.strip("\n").split("|")
    if len(items) != 2:
        continue
    # 有别名
    if items[1] != "":
        a_words = items[1].split(',')
        for a in a_words:
            if a.lower() not in simplified_head:
                simplified_head[a.lower()] = [items[0].lower()]
            else:
                if items[0].lower() in simplified_head[a.lower()]:  # 有可能别名表里有重复的别名或仅大小写有区别的别名
                    continue
                else:
                    simplified_head[a.lower()].append(items[0].lower())

index_names = set()  # 实体词entity set
for line in open(os.path.join('preprocess/', 'formatted_question.txt'), 'r', encoding='utf-8'):
    items = line.strip().split("\t")
    if len(items) != 5:
        continue
    head_entity = items[0].lower()
    tail_entity = items[2].lower()
    index_names.add(head_entity)
    index_names.add(tail_entity)


######################## Entity Detection  ########################
# 词库一定要与训练时完全统一
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

index2tag = np.array(ED.vocab.itos)
# print(index2tag)
idxO = int(np.where(index2tag == 'O')[0][0])  # Index for 'O'  训练集dete_train.txt中的标记
idxI = int(np.where(index2tag == 'I')[0][0])  # Index for 'I'
index2word = np.array(TEXT.vocab.itos)
# run the model on the test set and write the output to a file

################################################################
while True:
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

    # input_question = "who recorded the song shake ya body ?"
    input_question = input("plz input your question: ")
    input_question = input_question.strip()
    if input_question == '':  # 防止输入空白
        print('DO NOT INPUT BLANK QUESTION!!!')
        print('---------------------------------------------------------')
        continue
    input_question = input_question.lower()  # 转换为小写

    # 去除英文标点符号
    punctuation_en = string.punctuation  # punctuation: 标点符号
    for i in punctuation_en:
        if i not in ('(', ')', '-', "'", '&'):
            input_question = input_question.replace(i, '')

    if input_question.find('  ') != -1:
        input_question = input_question.replace('  ', ' ')  # 让间隔都为单间隔

    input_question_list = input_question.split(' ')

    outfile = open('try.txt', "w", encoding='utf-8')
    outfile.write(input_question)
    outfile.close()

    input_question = data.TabularDataset(path='try.txt', format='tsv', fields=[('text', TEXT)])

    if args.gpu == -1:  # Load all tensors onto the CPU
        input_question_iter = data.Iterator(input_question, batch_size=1, train=False, repeat=False, sort=False, shuffle=False, sort_within_batch=False)
    else:
        input_question_iter = data.Iterator(input_question, batch_size=1, device=torch.device('cuda', args.gpu), train=False,
                                            repeat=False, sort=False, shuffle=False, sort_within_batch=False)

    # print(get_entity(input_question))
    dete_result = get_entity(dataset_iter=input_question_iter)

    # pred_span = get_span(dete_result[0], index2tag, type=False)

    del model

    ######################## 处理之前 question 提取出的 head  ########################
    # 处理之前 question 提取出的 head   ###########################
    head_mid_idx = []  # [head1,head2,...]

    whhowset = [{'what', 'how', 'where', 'who', 'which', 'whom'},
                {'in which', 'what is', "what 's", 'what are', 'what was', 'what were', 'where is', 'where are',
                 'where was', 'where were', 'who is', 'who was', 'who are', 'how is', 'what did'},
                {'what kind of', 'what kinds of', 'what type of', 'what types of', 'what sort of'}]
    dete_tokens_list, filter_q = [], []
    # 下面开始处理之前提取出的 head
    question = input_question_list  # ['what', 'city', 'was', 'alex', 'golfis', 'born', 'in']
    pred_span = get_span(dete_result[0], index2tag, type=False)
    tokens_list, dete_tokens, st, en, changed = [], [], 0, 0, 0
    for st, en in pred_span:  # 依次挑出 一个question里 所有的 head
        tokens = question[st:en]
        print('the detected head is : ')
        print(tokens)
        print('----')
        tokens_list.append(tokens)  # [['alex', 'golfis']]
        if ' '.join(tokens) in index_names:  # important
            dete_tokens.append(' '.join(tokens))  # 探测出的, 且在之前txt出现的 head  ['alex golfis']
            head_mid_idx.append(' '.join(tokens))  # 探测出的, 且在之前txt出现的 head  [heads in question]

    if not head_mid_idx:  # 如果没有匹配成功的 head  先尝试用户是不是用了简写
        for st, en in pred_span:
            tokens = question[st:en]
            word = ' '.join(tokens)
            if word in simplified_head:
                for e in simplified_head[word]:
                    dete_tokens.append(e)
                    head_mid_idx.append(e)


    # ######################################################################
    # n-gram
    original_q = question.copy()  # 给后面的fuzzy使用
    if len(question) > 2:
        for j in range(3, 0, -1):
            if ' '.join(question[0:j]) in whhowset[j - 1]:  # 穷举问句前缀长度
                changed = j  # 找到问句前缀的长度 changed
                del question[0:j]  # 删去问句前缀
                continue
    tokens_list.append(question)  # [['alex', 'golfis'], ['city', 'was', 'alex', 'golfis', 'born', 'in']]
    filter_q.append(' '.join(question[:st - changed] + question[en - changed:]))  # filter_q 是 删去head，问句前缀 的 剩下句子

    if not head_mid_idx:  # 如果没有匹配成功的 head
        dete_tokens = question  # 如果没有匹配成功的 head, 就把除 what等疑问词 之外的所有句子加进dete_tokens_list
        for tokens in tokens_list:  # 遍历所有没匹配的head   包括除问句前缀的单词组成的tokens['city', 'was', 'alex', 'golfis', 'born', 'in']
            grams = []  # 存储 原先匹配不成功的 head 的子字符串
            maxlen = len(tokens)
            # 尝试扩展原先匹配不成功的 head, 穷举其子字符串
            for j in range(maxlen - 1, 1, -1):
                for token in [tokens[idx:idx + j] for idx in range(maxlen - j + 1)]:
                    grams.append(' '.join(token))
            for gram in grams:  # 原先匹配不成功的 head 的子字符串  是否  为数据库里的head
                if gram in index_names:
                    if gram not in head_mid_idx:  # 避免重复写入
                        head_mid_idx.append(gram)
                    break
                elif gram in simplified_head:  # 尝试缩写
                    for e in simplified_head[gram]:
                        if e not in head_mid_idx:  # 避免重复写入
                            head_mid_idx.append(e)
                    break
            # 接下来尝试删去某个肯定为非head的单词  再用剩下的tokens看是不是匹配数据库里的head
            for j, token in enumerate(tokens):
                if token not in match_pool:  # match_pool 存储   relation 实体词 + stopwords + 's
                    tokens = tokens[j:]
                    break
            if ' '.join(tokens) in index_names:
                if ' '.join(tokens) not in head_mid_idx:  # 避免重复写入
                    head_mid_idx.append(' '.join(tokens))
            elif ' '.join(tokens) in simplified_head:  # 尝试缩写
                for e in simplified_head[' '.join(tokens)]:
                    if e not in head_mid_idx:  # 避免重复写入
                        head_mid_idx.append(e)
            tokens = tokens[::-1]  # 反向再来一次
            for j, token in enumerate(tokens):
                if token not in match_pool:
                    tokens = tokens[j:]
                    break
            tokens = tokens[::-1]
            if ' '.join(tokens) in index_names:
                if ' '.join(tokens) not in head_mid_idx:  # 避免重复写入
                    head_mid_idx.append(' '.join(tokens))
            elif ' '.join(tokens) in simplified_head:  # 尝试缩写
                for e in simplified_head[' '.join(tokens)]:
                    if e not in head_mid_idx:  # 避免重复写入
                        head_mid_idx.append(e)

    # ######################################################################3
    if not head_mid_idx:  # 如果没有匹配成功的 head  尝试使用fuzzy匹配
        dete_tokens = []
        for st, en in pred_span:
            tokens = original_q[st:en]
            word = ' '.join(tokens)
            # # 去除英文标点符号
            # punctuation_en = string.punctuation  # punctuation: 标点符号
            # for i in punctuation_en:
            #     word = word.replace(i, '')
            if word.find('  ') != -1:
                word = word.replace('  ', ' ')  # 让间隔都为单间隔

            f = diy_fuzzy()  # 使用自己编写的 fuzzy 方法
            res = f.get_ent_from_fuzzy(word)  # 可能得到的是缩写, 所以还要将缩写转换为一般entity
            # # 返回最高得分的string
            # if res:  # 如果找得出大于阈值的答案
            #     if res in simplified_head:
            #         for e in simplified_head[res]:  # 可能一个别名对应多个实体
            #             dete_tokens.append(e)
            #             head_mid_idx.append(e)
            #     else:
            #         dete_tokens.append(res)
            #         head_mid_idx.append(res)

            # 返回所有score高于threshold的string list
            if len(res) > 0:
                for r in res:
                    if r in simplified_head:
                        for e in simplified_head[r]:  # 可能一个别名对应多个实体
                            dete_tokens.append(e)
                            head_mid_idx.append(e)
                    else:
                        dete_tokens.append(r)
                        head_mid_idx.append(r)


    # 最后决定的head是
    print('the preprocessed head is : ')
    print(head_mid_idx)
    print('----')


    # 手动改变 head_mid_idx, 尝试得到 distance threshold 的最佳取值
    # head_mid_idx = ['Faculty of Business']
    # dete_tokens = ['Faculty of Business']

    # 如果还是没有探测出 head entity, 则不用再往后进行 KGQA
    if not head_mid_idx:
        print('KGQA could not answer this question (could not detect the head entity), plz use content-based QA!!!')
        print('---------------------------------------------------------')
        continue

    dete_tokens_list.append(' '.join(dete_tokens))  # ['alex golfis', 'a tv action show ?',  ...]
    # 区分于 head_mid_idx ['alex golfis', '?', ...]


    id_match = False  # 如果提取出了 head 在 head_mid_idx, 设置为 True
    match_mid_list = []  # 按顺序存储 headid   ['m.0wzc58l', 'm.0jtw9c', 'm.0gys2sn', 'm.01fwty', 'm.0598nkm', ...]
    tupleset = []  # ('m.0wzc58l', 'alex golfis'), ('m.0jtw9c', 'phil hay'), ('m.0gys2sn', 'roger marquis'), ('m.01fwty', 'yves klein'), ('m.0598nkm', 'yves klein'), ...]
    # 对提取出的 head 进行匹配id
    tuplelist = []
    for name in head_mid_idx:  # 对一个 question 里的每个 head
        # match_mid_list.extend(name)  # 存储 id  可能存在一个 head 对应多种不同 id 的情况
        match_mid_list.append(name)  # 因为 name 改成了 字符串, 所以要改用 append
        if mid_dic.get(name) is not None:
            tuplelist.append(name)
    tupleset.extend(tuplelist)
    head_mid_idx = list(set(tuplelist))  # n-gram 可能会产生重复的 head entity, 需要用set去重

    if tuplelist:
        id_match = True
    tupleset = set(tupleset)
    # 从 'data/FB5M.name.txt' 取出所有 (entity id, entity 实体名) 信息
    tuple_topic = []  # (entity id, entity 实体名)
    for i in tupleset:
        # entity_id, entity_name = i
        # if entity_id in FB5M_id_to_name and FB5M_id_to_name[entity_id] == entity_name:
        #     tuple_topic.append(i)
        if i in mid_dic:  #
            tuple_topic.append(i)
    tuple_topic = set(tuple_topic)


    ######################## Learn entity representation  ########################
    # load the model
    if args.gpu == -1:  # Load all tensors onto the CPU
        model = torch.load(args.entity_model, map_location=lambda storage, loc: storage)
        model.config.cuda = False
    else:
        model = torch.load(args.entity_model, map_location=lambda storage, loc: storage.cuda(args.gpu))

    model.eval()
    for data_batch_idx, data_batch in enumerate(input_question_iter):
        scores = model(data_batch).cpu().data.numpy()
        break
    head_emb = scores[0]
    # print(head_emb)
    del model


    ######################## Learn predicate representation  ########################
    # load the model
    if args.gpu == -1:  # Load all tensors onto the CPU
        model = torch.load(args.pred_model, map_location=lambda storage, loc: storage)
        model.config.cuda = False
    else:
        model = torch.load(args.pred_model, map_location=lambda storage, loc: storage.cuda(args.gpu))

    model.eval()
    for data_batch_idx, data_batch in enumerate(input_question_iter):
        scores = model(data_batch).cpu().data.numpy()
        break
    pred_emb = scores[0]
    # print(pred_emb)
    del model


    ######################## predict and evaluation ########################
    # print('the match_mid_list is : ')
    # print(set(match_mid_list))
    # print('--')
    reach_dic = compute_reach_dic(set(match_mid_list))  # 返回 reach_dic[head_name] = [pred idx 1, pred idx 2, ...]
    # print('the reach_dic is : ')
    # print(reach_dic)
    # print('--')
    learned_pred, learned_fact, learned_head = [], set(), []  # 这些只会产生一个, 所以可以不用使用数组和set
    # 开始计算 距离
    alpha1, alpha3 = .39, .43
    answers = []
    for head_name in head_mid_idx:
        mid_score = np.sqrt(np.sum(np.power(entities_emb[mid_dic[head_name]] - head_emb, 2)))  # 预测 head representation 和 实际 head representation 的距离
        #if name is None and head_id in names_map:
        #    name = names_map[head_id]
        name_score = - .003 * fuzz.ratio(head_name, dete_tokens_list[0])
        if head_name in tuple_topic:  # tuple_topic 存储 来自 FB5M.name 的 (entity, entity实体名)  (个人感觉有点重复, 因为 tuple_topic 以及是用 (head_id, name) 提取出来的)
            name_score -= .18
        if reach_dic.get(head_name) is not None:
            for pred_idx in reach_dic[head_name]:  # reach_dic[head_id] = pred_idx are numbers
                # rel_names = - .017 * fuzz.ratio(pre_num_dic[pred_idx].replace('.', ' ').replace('_', ' '), filter_q[0]) #0.017
                rel_names = - .017 * fuzz.ratio(pre_num_dic[pred_idx], filter_q[0])  # 0.017
                rel_score = np.sqrt(np.sum(np.power(predicates_emb[pred_idx] - pred_emb, 2))) + rel_names  # 预测 pred representation 和 实际 pred representation 的距离
                tai_score = np.sqrt(np.sum(
                    np.power(predicates_emb[pred_idx] + entities_emb[mid_dic[head_name]] - head_emb - pred_emb, 2)))  # 计算 tail 的距离
                # (预测出的headid, 用预测出的headid匹配出的tailidx, 距离)
                answers.append((head_name, pred_idx, alpha1 * mid_score + rel_score + alpha3 * tai_score + name_score, tai_score))
    # print(answers)
    if answers:
        answers.sort(key=lambda x: x[2])  # 排序, 选出距离最小的
        if answers[0][3] <= (1.41319 * 2):  # 最短 tail 的距离应该小于数据库每两个tail平均距离的1/n
            learned_head = answers[0][0]
            learned_pred = answers[0][1]
            learned_fact.add(' '.join([learned_head, pre_num_dic[learned_pred]]))  # 距离最短的[head, pred]
        # learned_head = answers[0][0]
        # learned_pred = answers[0][1]
        # learned_fact.add(' '.join([learned_head, pre_num_dic[learned_pred]]))  # 距离最短的[head, pred]

    print('the nearest head and pred are: ')
    print(learned_fact)
    print('----')
    # 根据最短的[headid, predid], 从 'cleanedFB.txt' 找到 tail
    learned_tail = []
    if learned_fact:
        for line in open(os.path.join('preprocess/', 'formatted_question.txt'), 'r', encoding='utf-8'):
            items = line.strip().split("\t")
            # 根据 预测出来最好的[head, pred] 从 'formatted_question.txt' 取出 tail
            if ' '.join([items[0].lower(), items[1].lower()]) in learned_fact:
                # learned_tail.extend(items[2].split())
                learned_tail.append(items[2])
                # print(items)
                break  # 减少运行时间

    print('the answer is: ')
    if learned_tail:
        print(learned_tail)
    else:
        print('KGQA could not answer this question (the nearest triple is still too far), plz use content-based QA!!!')
    print('---------------------------------------------------------')


    for line in fileinput.input('try.txt', inplace=True):
        if not fileinput.isfirstline():
            print(line.replace('\n', ''))