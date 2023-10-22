import torch
import numpy as np
import random
import os
import fileinput
import json
import string
from nltk.corpus import stopwords
# from itertools import compress
from KGQA.evaluation import evaluation, get_span
# from evaluation import evaluation, get_span
# from argparse import ArgumentParser
# from torchtext import data
from torchtext.legacy import data
# from sklearn.metrics.pairwise import euclidean_distances
from fuzzywuzzy import fuzz
# from util import www2fb, processed_text, clean_uri
from KGQA.Args import Args
from KGQA.EntityLinking import EntityLinking

# from Args import Args

class KGQA:
    def __init__(self):
        self.args = Args()
        self.cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.head_mid_idx = []
        self.dete_tokens_list = []
        self.filter_q = []
        self.head = ""
        self.relation = ""

        # Set random seed for reproducibility
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

        if not self.args.cuda:
            self.args.gpu = -1
        if torch.cuda.is_available() and self.args.cuda:
            print("Note: You are using GPU for testing")
            torch.cuda.set_device(self.args.gpu)
            torch.cuda.manual_seed(self.args.seed)
        if torch.cuda.is_available() and not self.args.cuda:
            print("Warning: You have Cuda but not use it. You are using CPU for testing.")
        if not torch.cuda.is_available() and not self.args.cuda:
            print("Warning: You are using CPU for testing.")

        ######################## Prepare some dictionary  ########################
        # 构建 entity:entityidx  entityidx:entity  pred:predidx  predidx:pred  的字典  ###########################
        self.mid_dic, mid_num_dic = {}, {}  # Dictionary for MID     entity:entityidx
        for line in open(os.path.join(self.cur_dir, 'preprocess/entity2id.txt'), 'r', encoding='utf-8'):
            items = line.strip().split("|")
            self.mid_dic[items[0].lower()] = int(items[1])
            mid_num_dic[int(items[1])] = items[0].lower()
        self.pre_dic, self.pre_num_dic = {}, {}  # Dictionary for predicates   pred:predidx
        self.match_pool = []  # 存 relation 实体词
        for line in open(os.path.join(self.cur_dir, 'preprocess/relation2id.txt'), 'r', encoding='utf-8'):
            items = line.strip().split("|")
            self.match_pool = self.match_pool + items[0].lower().split()
            self.pre_dic[items[0].lower()] = int(items[1])
            self.pre_num_dic[int(items[1])] = items[0].lower()
        # 加入 stopwords
        self.match_pool = set(self.match_pool + stopwords.words('english') + ["'s"])

        # Embedding for MID
        n = 0
        for line in open(os.path.join(self.cur_dir, 'preprocess/entity_250dim.txt'), 'r', encoding='utf-8'):
            items = line.strip().split("\t")
            if len(items) != 2:
                print(items)
                continue
            n += 1

        arr = [[] for _ in range(n)]
        for line in open(os.path.join(self.cur_dir, 'preprocess/entity_250dim.txt'), 'r', encoding='utf-8'):
            items = line.strip().split("\t")
            if len(items) != 2:
                print(items)
                continue
            # 现在 items[1] 是一堆字符串, 需要处理
            embedding = json.loads(items[1])
            arr[int(items[0])] = embedding  # idx 对应原本 entity 的编号
            # arr.append(items[1])
        self.entities_emb = np.array(arr, dtype=np.float32)

        n = 0
        for line in open(os.path.join(self.cur_dir, 'preprocess/relation_250dim.txt'), 'r', encoding='utf-8'):
            items = line.strip().split("\t")
            if len(items) != 2:
                print(items)
                continue
            n += 1
        arr = [[] for _ in range(n)]
        for line in open(os.path.join(self.cur_dir, 'preprocess/relation_250dim.txt'), 'r', encoding='utf-8'):
            items = line.strip().split("\t")
            if len(items) != 2:
                print(items)
                continue
            # 现在 items[1] 是一堆字符串, 需要处理
            embedding = json.loads(items[1])
            arr[int(items[0])] = embedding  # idx 对应原本 entity 的编号
            # arr.append(items[1])
        self.predicates_emb = np.array(arr, dtype=np.float32)

        # 构建 简写head:head 的dictionary
        self.simplified_head = dict()
        for line in open(os.path.join(self.cur_dir, 'preprocess/alias.txt'), 'r', encoding='utf-8'):
            items = line.strip("\n").split("|")
            if len(items) != 2:
                continue
            # 有别名
            if items[1] != "":
                a_words = items[1].split(',')
                for a in a_words:
                    if a.lower() == items[0].lower():  # 有可能别名表里有重复的别名或仅大小写有区别的别名
                        continue
                    if a.lower() not in self.simplified_head:
                        self.simplified_head[a.lower()] = [items[0].lower()]
                    else:
                        if items[0].lower() in self.simplified_head[a.lower()]:  # 有可能别名表里有重复的别名或仅大小写有区别的别名
                            continue
                        else:
                            self.simplified_head[a.lower()].append(items[0].lower())

        self.index_names = set()  # 实体词entity set
        for line in open(os.path.join(self.cur_dir, 'preprocess/formatted_question.txt'), 'r', encoding='utf-8'):
            items = line.strip().split("\t")
            if len(items) != 5:
                continue
            head_entity = items[0].lower()
            tail_entity = items[2].lower()
            self.index_names.add(head_entity)
            self.index_names.add(tail_entity)

        ######################## Entity Detection  ########################
        # 词库一定要与训练时完全统一
        self.TEXT = data.Field(lower=True)
        ED = data.Field()
        #  75710 ############################################################  把 polyu 问题手动cv到 dete_train.txt
        train = data.TabularDataset(path=os.path.join(self.cur_dir, 'freebase/dete_train.txt'), format='tsv',
                                    fields=[('text', self.TEXT), ('ed', ED)])
        # 设置为 none 则丢弃
        field = [('id', None), ('sub', None), ('entity', None), ('relation', None), ('obj', None), ('text', self.TEXT),
                 ('ed', ED)]
        dev, test = data.TabularDataset.splits(path=os.path.join(self.cur_dir, 'freebase/'), validation='valid.txt',
                                               test='test.txt', format='tsv',
                                               fields=field)
        polyu_dataset = data.TabularDataset(path=os.path.join(self.cur_dir, 'preprocess/formatted_question.txt'),
                                            format='tsv',
                                            fields=[('entity', None), ('relation', None), ('obj', None),
                                                    ('text', self.TEXT),
                                                    ('ed', None)])
        polyu_dataset_dev = data.TabularDataset(
            path=os.path.join(self.cur_dir, 'preprocess/whole_tabbed_question_only.txt'),
            format='tsv', fields=[('text', self.TEXT), ('ed', ED)])
        self.TEXT.build_vocab(train, dev, test, polyu_dataset,
                              polyu_dataset_dev)  # 构建词表时要加入polyu相关词 ######################################
        ED.build_vocab(train, dev, polyu_dataset_dev)

        # total_num = len(test)
        # print('total num of example: {}'.format(total_num))

        self.index2tag = np.array(ED.vocab.itos)
        # print(index2tag)
        # idxO = int(np.where(self.index2tag == 'O')[0][0])  # Index for 'O'  训练集dete_train.txt中的标记
        # idxI = int(np.where(self.index2tag == 'I')[0][0])  # Index for 'I'
        # index2word = np.array(self.TEXT.vocab.itos)
        # run the model on the test set and write the output to a file

        ############################load the head entity model####################################
        # load the model
        if self.args.gpu == -1:  # Load all tensors onto the CPU
            # test_iter = data.Iterator(test, batch_size=args.batch_size, train=False, repeat=False, sort=False, shuffle=False,
            #                           sort_within_batch=False)
            self.model = torch.load(self.args.dete_model, map_location=lambda storage, loc: storage)
            self.model.config.cuda = False
        else:
            # test_iter = data.Iterator(test, batch_size=args.batch_size, device=torch.device('cuda', args.gpu), train=False,
            #                           repeat=False, sort=False, shuffle=False, sort_within_batch=False)
            self.model = torch.load(self.args.dete_model, map_location=lambda storage, loc: storage.cuda(self.args.gpu))

        ######################## load entity representation  ########################
        # load the model
        if self.args.gpu == -1:  # Load all tensors onto the CPU
            self.entity_representation_model = torch.load(self.args.entity_model,
                                                          map_location=lambda storage, loc: storage)
            self.entity_representation_model.config.cuda = False
        else:
            self.entity_representation_model = torch.load(self.args.entity_model,
                                                          map_location=lambda storage, loc: storage.cuda(self.args.gpu))

        ######################## Learn predicate representation  ########################
        # load the model
        if self.args.gpu == -1:  # Load all tensors onto the CPU
            self.predicate_representation_model = torch.load(self.args.pred_model,
                                                             map_location=lambda storage, loc: storage)
            self.predicate_representation_model.config.cuda = False
        else:
            self.predicate_representation_model = torch.load(self.args.pred_model,
                                                             map_location=lambda storage, loc: storage.cuda(
                                                                 self.args.gpu))

        print("Knowledge graph model init finished ......")

    def triple_to_answer(self, t):  # 输入为 1个triple , 返回一个句子
        pre = t[1]  # 取出 predicate
        if pre == 'Website':
            return "The website of " + t[0] + " is  " + t[2]
        elif pre == 'Professional Title':
            return "The professional title of " + t[0] + " is  " + t[2]
        elif pre == 'Office':
            return "The office of " + t[0] + " is  " + t[2]
        elif pre == 'Telephone':
            return "The telephone of " + t[0] + " is  " + t[2]
        elif pre == 'Email':
            return "The email address of " + t[0] + " is  " + t[2]
        elif pre == 'Credits Required for Graduation':
            return "Credits required for graduation for " + t[0] + " is  " + t[2]
        elif pre == 'Department':
            return "The department of " + t[0] + " is  " + t[2]
        elif pre == 'Duration':
            return "The duration of " + t[0] + " is  " + t[2]
        elif pre == 'Students Number':
            return "The students number in " + t[0] + " is  " + t[2]
        elif pre == 'Graduates Number':
            return "The graduates number in " + t[0] + " is  " + t[2]
        elif pre == 'Faculty Members Number':
            return "The faculty members number in " + t[0] + " is  " + t[2]
        elif pre == 'Taught Programmes Number':
            return "The taught programmes number in " + t[0] + " is  " + t[2]
        elif pre == 'Date of Creation':
            return "The date of creation of " + t[0] + " is:  " + t[2]
        elif pre == 'Sports Facilities':
            return "There are the following Sports Facilities in " + t[0] + ":  " + t[2]
        elif pre == 'Student Halls of Residence':
            return "There are the following Student Halls of Residence in " + t[0] + ":  " + t[2]
        elif pre == 'Catering':
            return "There are the following Caterings in " + t[0] + ":  " + t[2]
        elif pre == 'Health Service':
            return "There are the following Health Services in " + t[0] + ":  " + t[2]
        elif pre == 'Global Student Hub':
            return "There are the following Global Student Hubs in " + t[0] + ":  " + t[2]
        elif pre == 'Student Lockers':
            return "There are the following Student Lockers in " + t[0] + ":  " + t[2]
        elif pre == 'Library':
            return "The Library in " + t[0] + " is  " + t[2]
        elif pre == 'Faculty':
            return "There are the following Faculties in " + t[0] + ":  " + t[2]
        elif pre == 'School':
            return "There are the following Schools in " + t[0] + ":  " + t[2]
        elif pre == 'Location':
            return "The location of " + t[0] + " is  " + t[2]
        elif pre == 'Mode of Study':
            return "The mode of study for " + t[0] + " is  " + t[2]
        elif pre == 'Ranking':
            return "The ranking of " + t[0] + " is  " + t[2]
        elif pre == 'Taught Postgraduate Programmes':
            return "There are the following Taught Postgraduate Programmes in " + t[0] + ":  " + t[2]
        elif pre == 'Fund Type':
            return "The fund type of " + t[0] + " is  " + t[2]
        elif pre == 'Compulsory Subjects':
            return "The Compulsory Subjects in " + t[0] + " are:  " + t[2]
        elif pre == 'Elective Subjects':
            return "The Elective Subjects in " + t[0] + " are:  " + t[2]
        elif pre == 'Tuition':
            return "The tuition fee of " + t[0] + " is:  " + t[2]
        elif pre == 'Application':
            return "There are these real-world applications of " + t[0] + ":  " + t[2]
        elif pre == 'Book':
            return "You can read this book to learn " + t[0] + ":  " + t[2]
        elif pre == 'Field of Work':
            return "The Field of Works of " + t[0] + " are:  " + t[2]
        elif pre == 'Introduction':
            return t[2]
        elif pre == 'Inventor':
            return "The inventor of " + t[0] + " is:  " + t[2]
        elif pre == 'Subtopic':
            return t[0] + " has the following Subtopics:  " + t[2]
        elif pre == 'Supertopic':
            return t[0] + "'s Supertopics are:  " + t[2]
        elif pre == 'Video':
            return "You can learn " + t[0] + " by watching this video:  " + t[2]
        else:
            return t[2]


    def compute_reach_dic(self, matched_mid):  # 输入为 所有 matched 的 head_name
        reach_dic = {}
        with open(os.path.join(self.cur_dir, 'preprocess/formatted_question.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                items = line.strip().split("\t")
                head_name = items[0].lower()  # 记得转换为小写
                if head_name in matched_mid and items[1].lower() in self.pre_dic:
                    if reach_dic.get(head_name) is None:
                        reach_dic[head_name] = [self.pre_dic[items[1].lower()]]
                    else:
                        reach_dic[head_name].append(self.pre_dic[items[1].lower()])
        return reach_dic  # 返回 reach_dic[head_name] = [pred idx 1, pred idx 2, ...]

    def get_head_entity(self, input_question):
        ######################## 问题处理  ########################
        input_question = input_question.strip()  # 去除前后空格
        input_question = input_question.lower()  # 转换为小写
        # 去除英文标点符号
        # punctuation_en = string.punctuation  # punctuation: 标点符号
        # for i in punctuation_en:
        #     if i not in ('(', ')', '-', "'", '&', ',', ':', '.', '/', '+'):
        #         input_question = input_question.replace(i, '')
        # 让间隔都为单间隔
        if input_question.find('  ') != -1:
            input_question = input_question.replace('  ', ' ')
        # 把问题根据空格变成tokens list
        input_question_list = input_question.split(' ')

        # 将问题转化为torch可处理格式
        outfile = open('try.txt', "w", encoding='utf-8')
        outfile.write(input_question)
        outfile.close()
        input_question = data.TabularDataset(path='try.txt', format='tsv', fields=[('text', self.TEXT)])
        # 是否读入CPU
        if self.args.gpu == -1:  # Load all tensors onto the CPU
            input_question_iter = data.Iterator(input_question, batch_size=1, train=False, repeat=False, sort=False,
                                                shuffle=False, sort_within_batch=False)
        else:
            input_question_iter = data.Iterator(input_question, batch_size=1,
                                                device=torch.device('cuda', self.args.gpu),
                                                train=False,
                                                repeat=False, sort=False, shuffle=False, sort_within_batch=False)
        # 模型评估开启
        self.model.eval()
        input_question_iter.init_epoch()
        dete_result = []
        for data_batch_idx, data_batch in enumerate(input_question_iter):
            # batch_size = data_batch.text.size()[1]
            answer = torch.max(self.model(data_batch), 1)[1].view(data_batch.text.size())
            answer[(data_batch.text.data == 1)] = 1
            answer = np.transpose(answer.cpu().data.numpy())
            dete_result.extend(answer)
            break

        ######################## 处理之前 question 提取出的 head  ########################
        # 处理之前 question 提取出的 head
        self.head_mid_idx = []  # [head1,head2,...]

        whhowset = [{'what', 'how', 'where', 'who', 'which', 'whom'},
                    {'in which', 'what is', "what 's", 'what are', 'what was', 'what were', 'where is', 'where are',
                     'where was', 'where were', 'who is', 'who was', 'who are', 'how is', 'what did'},
                    {'what kind of', 'what kinds of', 'what type of', 'what types of', 'what sort of'}]
        self.dete_tokens_list, self.filter_q = [], []
        # 下面开始处理之前提取出的 head
        question = input_question_list  # ['what', 'city', 'was', 'alex', 'golfis', 'born', 'in']
        # 获取head entity的(start,end)，可能有多个
        pred_span = get_span(dete_result[0], self.index2tag, type=False)
        tokens_list, dete_tokens, st, en, changed = [], [], 0, 0, 0

        for st, en in pred_span:  # 依次挑出一个question里所有的 head
            tokens = question[st:en]
            tokens_list.append(tokens)  # [['alex', 'golfis']]
            if ' '.join(tokens) in self.index_names:  # important
                dete_tokens.append(' '.join(tokens))  # 探测出的, 且在之前txt出现的 head  ['alex golfis']
                self.head_mid_idx.append(' '.join(tokens))  # 探测出的, 且在之前txt出现的 head  [heads in question]

        # 尝试使用简写匹配
        if not self.head_mid_idx:  # 如果没有匹配成功的 head  先尝试用户是不是用了简写
            for st, en in pred_span:
                tokens = question[st:en]
                word = ' '.join(tokens)
                if word in self.simplified_head:
                    for e in self.simplified_head[word]:
                        dete_tokens.append(e)
                        self.head_mid_idx.append(e)

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
        self.filter_q.append(
            ' '.join(question[:st - changed] + question[en - changed:]))  # filter_q 是 删去head，问句前缀 的 剩下句子

        if not self.head_mid_idx:  # 如果没有匹配成功的 head
            dete_tokens = question  # 如果没有匹配成功的 head, 就把除 what等疑问词 之外的所有句子加进dete_tokens_list
            for tokens in tokens_list:  # 遍历所有没匹配的head   包括除问句前缀的单词组成的tokens['city', 'was', 'alex', 'golfis', 'born', 'in']
                grams = []  # 存储 原先匹配不成功的 head 的子字符串
                maxlen = len(tokens)
                # 尝试扩展原先匹配不成功的 head, 穷举其子字符串
                for j in range(maxlen - 1, 1, -1):
                    for token in [tokens[idx:idx + j] for idx in range(maxlen - j + 1)]:
                        grams.append(' '.join(token))
                for gram in grams:  # 原先匹配不成功的 head 的子字符串  是否  为数据库里的head
                    if gram in self.index_names:
                        if gram not in self.head_mid_idx:  # 避免重复写入
                            self.head_mid_idx.append(gram)
                        break
                    elif gram in self.simplified_head:  # 尝试缩写
                        for e in self.simplified_head[gram]:
                            if e not in self.head_mid_idx:  # 避免重复写入
                                self.head_mid_idx.append(e)
                        break
                # 接下来尝试删去某个肯定为非head的单词  再用剩下的tokens看是不是匹配数据库里的head
                for j, token in enumerate(tokens):
                    if token not in self.match_pool:  # match_pool 存储   relation 实体词 + stopwords + 's
                        tokens = tokens[j:]
                        break
                if ' '.join(tokens) in self.index_names:
                    if ' '.join(tokens) not in self.head_mid_idx:  # 避免重复写入
                        self.head_mid_idx.append(' '.join(tokens))
                elif ' '.join(tokens) in self.simplified_head:  # 尝试缩写
                    for e in self.simplified_head[' '.join(tokens)]:
                        if e not in self.head_mid_idx:  # 避免重复写入
                            self.head_mid_idx.append(e)
                tokens = tokens[::-1]  # 反向再来一次
                for j, token in enumerate(tokens):
                    if token not in self.match_pool:
                        tokens = tokens[j:]
                        break
                tokens = tokens[::-1]
                if ' '.join(tokens) in self.index_names:
                    if ' '.join(tokens) not in self.head_mid_idx:  # 避免重复写入
                        self.head_mid_idx.append(' '.join(tokens))
                elif ' '.join(tokens) in self.simplified_head:  # 尝试缩写
                    for e in self.simplified_head[' '.join(tokens)]:
                        if e not in self.head_mid_idx:  # 避免重复写入
                            self.head_mid_idx.append(e)

        # ######################################################################3
        if not self.head_mid_idx:  # 如果没有匹配成功的 head  尝试使用fuzzy匹配
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

                f = EntityLinking()  # 使用自己编写的 fuzzy 方法
                candidate_entities = f.entity_link(word)  # 不管是缩写还是全称, 都统一转换为了一般entity
                if len(candidate_entities) > 0:  # 如果找得出大于阈值的答案
                    for e in candidate_entities:
                        # 返回的 candidate_entities 可能为大写, 记得转换为小写
                        dete_tokens.append(e.lower())  # 探测出的, 且在之前txt出现的 head  ['alex golfis']
                        self.head_mid_idx.append(e.lower())  # 探测出的, 且在之前txt出现的 head  [heads in question]

        self.dete_tokens_list.append(' '.join(dete_tokens))
        # 最后决定的head是
        # print('the preprocessed head is : ')
        # print(self.head_mid_idx)
        # print('--')
        return self.head_mid_idx

    def get_answer(self):
        input_question = data.TabularDataset(path='try.txt', format='tsv', fields=[('text', self.TEXT)])
        if self.args.gpu == -1:
            input_question_iter = data.Iterator(input_question, batch_size=1, train=False, repeat=False, sort=False,
                                                shuffle=False, sort_within_batch=False)
        else:
            input_question_iter = data.Iterator(input_question, batch_size=1,
                                                device=torch.device('cuda', self.args.gpu),
                                                train=False,
                                                repeat=False, sort=False, shuffle=False, sort_within_batch=False)

        id_match = False  # 如果提取出了 head 在 head_mid_idx, 设置为 True
        match_mid_list = []  # 按顺序存储 headid   ['m.0wzc58l', 'm.0jtw9c', 'm.0gys2sn', 'm.01fwty', 'm.0598nkm', ...]
        tupleset = []  # ('m.0wzc58l', 'alex golfis'), ('m.0jtw9c', 'phil hay'), ('m.0gys2sn', 'roger marquis'), ('m.01fwty', 'yves klein'), ('m.0598nkm', 'yves klein'), ...]
        # 对提取出的 head 进行匹配id
        tuplelist = []
        for name in self.head_mid_idx:  # 对一个 question 里的每个 head
            # match_mid_list.extend(name)  # 存储 id  可能存在一个 head 对应多种不同 id 的情况
            match_mid_list.append(name)  # 因为 name 改成了 字符串, 所以要改用 append
            if self.mid_dic.get(name) is not None:
                tuplelist.append(name)
        tupleset.extend(tuplelist)
        self.head_mid_idx = list(set(tuplelist))  # [('m.0wzc58l', 'alex golfis'), ('m.0jtw9c', 'phil hay')], ...]
        if tuplelist:
            id_match = True
        tupleset = set(tupleset)
        # 从 'data/FB5M.name.txt' 取出所有 (entity id, entity 实体名) 信息
        tuple_topic = []  # (entity id, entity 实体名)
        for i in tupleset:
            if i in self.mid_dic:  #
                tuple_topic.append(i)
        tuple_topic = set(tuple_topic)

        self.entity_representation_model.eval()
        for data_batch_idx, data_batch in enumerate(input_question_iter):
            scores = self.entity_representation_model(data_batch).cpu().data.numpy()
            break
        head_emb = scores[0]

        self.predicate_representation_model.eval()
        for data_batch_idx, data_batch in enumerate(input_question_iter):
            scores = self.predicate_representation_model(data_batch).cpu().data.numpy()
            break
        pred_emb = scores[0]

        ######################## predict and evaluation ########################
        reach_dic = self.compute_reach_dic(
            set(match_mid_list))  # 返回 reach_dic[head_name] = [pred idx 1, pred idx 2, ...]
        # print('the reach_dic is : ')
        # print(reach_dic)
        # print('--')
        learned_pred, learned_fact, learned_head = [], set(), []
        # 开始计算 距离
        alpha1, alpha3 = .39, .43
        answers = []
        for head_name in self.head_mid_idx:
            mid_score = np.sqrt(np.sum(np.power(self.entities_emb[self.mid_dic[head_name]] - head_emb,
                                                2)))  # 预测 head representation 和 实际 head representation 的距离
            # if name is None and head_id in names_map:
            #    name = names_map[head_id]
            name_score = - .003 * fuzz.ratio(head_name, self.dete_tokens_list[0])
            if head_name in tuple_topic:  # tuple_topic 存储 来自 FB5M.name 的 (entity, entity实体名)  (个人感觉有点重复, 因为 tuple_topic 以及是用 (head_id, name) 提取出来的)
                name_score -= .18
            if reach_dic.get(head_name) is not None:
                for pred_idx in reach_dic[head_name]:  # reach_dic[head_id] = pred_idx are numbers
                    # rel_names = - .017 * fuzz.ratio(pre_num_dic[pred_idx].replace('.', ' ').replace('_', ' '), filter_q[0]) #0.017
                    # print(self.filter_q)
                    rel_names = - .017 * fuzz.ratio(self.pre_num_dic[pred_idx], self.filter_q[0])  # 0.017
                    rel_score = np.sqrt(np.sum(np.power(self.predicates_emb[pred_idx] - pred_emb,
                                                        2))) + rel_names  # 预测 pred representation 和 实际 pred representation 的距离
                    tai_score = np.sqrt(np.sum(
                        np.power(self.predicates_emb[pred_idx] + self.entities_emb[
                            self.mid_dic[head_name]] - head_emb - pred_emb,
                                 2)))  # 计算 tail 的距离
                    # (预测出的headid, 用预测出的headid匹配出的tailidx, 距离)
                    answers.append(
                        (head_name, pred_idx, alpha1 * mid_score + rel_score + alpha3 * tai_score + name_score, tai_score))
        # print(answers)
        if answers:
            answers.sort(key=lambda x: x[2])  # 排序, 选出距离最小的
            if answers[0][3] <= (1.4132385 * 1.5):  # 最短 tail 的距离应该小于数据库每两个tail平均距离的1/n
                learned_head = answers[0][0]
                learned_pred = answers[0][1]
                learned_fact.add(' '.join([learned_head, self.pre_num_dic[learned_pred]]))  # 距离最短的[head, pred]

        # print('the nearest head and pred are: ')
        # print(learned_fact)
        # print('--')
        # 根据最短的[headid, predid], 从 'cleanedFB.txt' 找到 tail
        # learned_tail = []


        res_arr = []
        if learned_fact:
            for line in open(os.path.join(self.cur_dir, 'preprocess/formatted_question.txt'), 'r', encoding='utf-8'):
                items = line.strip().split("\t")
                # 根据 预测出来最好的[head, pred] 从 'formatted_question.txt' 取出 tail
                if ' '.join([items[0].lower(), items[1].lower()]) in learned_fact:
                    # learned_tail.extend(items[2].split())
                    # learned_tail.append(items[2])
                    self.entity = items[0]
                    self.relation = items[1]
                    res_arr.append(self.triple_to_answer(items))  # 转换成 natural language 的句子输出
                    # print(items)
                    break  # 减少运行时间

            # print('the answer is: ')
            # print(learned_tail)
            # print('---------------------------------------------------------')
        if res_arr:
            if answers[0][3] <= (1.4132385 * 0.95):  # 距离小于 0.95*threshold, KGQA 大概率回答正确
                return "KGQA", res_arr[0]
            else:  # 距离处于模糊区间 (0.95*threshold, 1.50*threshold], KGQA 有概率回答错误, 可使用 content-based 多返回一个答案
                return "mix-mode", res_arr[0]
        else:  # 无法回答,直接使用content-based
            return "content-based", ""

if __name__ == '__main__':
    KGQA = KGQA()
    head_entity = KGQA.get_head_entity("where is pao library?")
    print(head_entity)
    answer = KGQA.get_answer()
    print(answer)
