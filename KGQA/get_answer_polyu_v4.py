import torch
import numpy as np
import random
import os
import fileinput
import json
import string

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

# 构建 简写head:head 的dictionary
simplified_head = dict()
for line in open(os.path.join('preprocess/', 'simplified_form.txt'), 'r', encoding='utf-8'):
    items = line.strip().split("\t")
    if len(items) != 2:
        continue
    simplified_head[items[0].lower()] = items[1].lower()

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
# 先构建词库
# TEXT = data.Field(lower=True)
# ED = data.Field()
# train = data.TabularDataset(path=os.path.join('freebase/', 'dete_train.txt'), format='tsv', fields=[('text', TEXT), ('ed', ED)])
# # 设置为 none 则丢弃
# field = [('id', None), ('sub', None), ('entity', None), ('relation', None), ('obj', None), ('text', TEXT), ('ed', ED)]
# dev, test = data.TabularDataset.splits(path='freebase/', validation='valid.txt', test='test.txt', format='tsv', fields=field)
# polyu_dataset = data.TabularDataset(path=os.path.join('preprocess/', 'formatted_question.txt'), format='tsv', fields=[('entity', None), ('relation', None), ('obj', None), ('text', TEXT), ('ed', ED)])
# TEXT.build_vocab(train, dev, test, polyu_dataset)  # 构建词表时要加入polyu相关词 ######################################
# ED.build_vocab(train, dev, polyu_dataset)
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
        print('the predicted head is : ')
        print(tokens)
        print('--')
        tokens_list.append(tokens)  # [['alex', 'golfis']]
        if ' '.join(tokens) in index_names:  # important
            dete_tokens.append(' '.join(tokens))  # 探测出的, 且在之前txt出现的 head  ['alex golfis']
            head_mid_idx.append(' '.join(tokens))  # 探测出的, 且在之前txt出现的 head  [heads in question]

    if not head_mid_idx:  # 如果没有匹配成功的 head  先尝试用户是不是用了简写
        for st, en in pred_span:
            tokens = question[st:en]
            word = ' '.join(tokens)
            if word in simplified_head:
                dete_tokens.append(simplified_head[word])  # 探测出的, 且在之前txt出现的 head  ['alex golfis']
                head_mid_idx.append(simplified_head[word])  # 探测出的, 且在之前txt出现的 head  [heads in question]


    # ######################################################################3
    if not head_mid_idx:  # 如果没有匹配成功的 head  尝试使用fuzzy匹配
        for st, en in pred_span:
            tokens = question[st:en]
            word = ' '.join(tokens)
            # 去除英文标点符号
            punctuation_en = string.punctuation  # punctuation: 标点符号
            for i in punctuation_en:
                word = word.replace(i, '')
            if word.find('  ') != -1:
                word = word.replace('  ', ' ')  # 让间隔都为单间隔

            res = []
            for line in open('preprocess/train.txt', 'r', encoding='utf-8'):  # 可以再提前读文件, 创建一个字典存好head entity, 节省时间
                items = line.strip().split("|")
                # head:items[0]  relation:items[1]  tail:items[2]
                if len(items) != 3:
                    continue
                # 此处还可以对数据库里的head也删除标点符号
                cur_head_lower = items[0].lower()
                res.append((0.3 * fuzz.ratio(word, cur_head_lower) +
                            0.3 * fuzz.partial_ratio(word, cur_head_lower) +
                            0.4 * fuzz.token_sort_ratio(word, cur_head_lower), cur_head_lower))  # 注意这里要放入小写
            res.sort(key=lambda x: x[0], reverse=True)
            # 此处可加入阈值, 大于该阈值再插入  阈值为res[0][0]
            dete_tokens.append(str(res[0][1]))  # 探测出的, 且在之前txt出现的 head  ['alex golfis']
            head_mid_idx.append(str(res[0][1]))  # 探测出的, 且在之前txt出现的 head  [heads in question]

    if len(question) > 2:  # 删去问题前缀
        for j in range(3, 0, -1):
            if ' '.join(question[0:j]) in whhowset[j - 1]:
                changed = j
                del question[0:j]
                continue
    tokens_list.append(question)  # [['alex', 'golfis'], ['city', 'was', 'alex', 'golfis', 'born', 'in']]
    filter_q.append(' '.join(question[:st - changed] + question[en - changed:]))  # 除去 question 最后一个 head,  ['city was born in', 'film is by the writer ?' ...]

    if not head_mid_idx:  # 如果没有匹配成功的 head
        dete_tokens = question  # 如果没有匹配成功的 head, 就把除 what等疑问词 之外的所有句子加进dete_tokens_list
        for tokens in tokens_list:
            grams = []
            maxlen = len(tokens)
            # 尝试扩展原先匹配不成功的 head, 穷举其周围的单词
            for j in range(maxlen - 1, 1, -1):
                for token in [tokens[idx:idx + j] for idx in range(maxlen - j + 1)]:
                    grams.append(' '.join(token))
            for gram in grams:
                if gram in index_names:
                    head_mid_idx.append(gram)
                    break
            for j, token in enumerate(tokens):
                if token not in match_pool:
                    tokens = tokens[j:]
                    break
            if ' '.join(tokens) in index_names:
                head_mid_idx.append(' '.join(tokens))
            tokens = tokens[::-1]
            for j, token in enumerate(tokens):
                if token not in match_pool:
                    tokens = tokens[j:]
                    break
            tokens = tokens[::-1]
            if ' '.join(tokens) in index_names:
                head_mid_idx.append(' '.join(tokens))


    # 最后决定的head是
    print('the preprocessed head is : ')
    print(head_mid_idx)
    print('--')

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
    head_mid_idx = list(set(tuplelist))  # [('m.0wzc58l', 'alex golfis'), ('m.0jtw9c', 'phil hay')], ...]
    if tuplelist:
        id_match = True
    tupleset = set(tupleset)
    # 从 'data/FB5M.name.txt' 取出所有 (entity id, entity 实体名) 信息
    tuple_topic = []  # (entity id, entity 实体名)
    # with open('data/FB5M.name.txt', 'r', encoding='utf-8') as f:
    #     for i, line in enumerate(f):
    #         # if i % 1000000 == 0:
    #         #     print("line: {}".format(i))
    #         items = line.strip().split("\t")
    #         if (www2fb(clean_uri(items[0])), processed_text(clean_uri(items[2]))) in tupleset and items[1] == "<fb:type.object.name>":
    #             tuple_topic.append((www2fb(clean_uri(items[0])), processed_text(clean_uri(items[2]))))
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
    learned_pred, learned_fact, learned_head = [], set(), []
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
                answers.append((head_name, pred_idx, alpha1 * mid_score + rel_score + alpha3 * tai_score + name_score))
    # print(answers)
    if answers:
        answers.sort(key=lambda x: x[2])  # 排序, 选出距离最小的
        learned_head = answers[0][0]
        learned_pred = answers[0][1]
        learned_fact.add(' '.join([learned_head, pre_num_dic[learned_pred]]))  # 距离最短的[head, pred]

    print('the nearest head and pred are: ')
    print(learned_fact)
    print('--')
    # 根据最短的[headid, predid], 从 'cleanedFB.txt' 找到 tail
    learned_tail = []
    for line in open(os.path.join('preprocess/', 'formatted_question.txt'), 'r', encoding='utf-8'):
        items = line.strip().split("\t")
        # 根据 预测出来最好的[head, pred] 从 'formatted_question.txt' 取出 tail
        if ' '.join([items[0].lower(), items[1].lower()]) in learned_fact:
            # learned_tail.extend(items[2].split())
            learned_tail.append(items[2])
            # print(items)
            break  # 减少运行时间

    print('the answer is: ')
    print(learned_tail)
    print('---------------------------------------------------------')
    # res = []
    # for res_tail in learned_tail:
    #     if res_tail in FB5M_id_to_name:
    #         res.append(FB5M_id_to_name[res_tail])
    #     else:
    #         continue
    # if len(res) > 0:
    #     print(res)
    # else:
    #     print("could not answer this question")

    for line in fileinput.input('try.txt', inplace=True):
        if not fileinput.isfirstline():
            print(line.replace('\n', ''))