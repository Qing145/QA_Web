import torch
import torch.nn as nn
import time
import os
import numpy as np
import random
import json

# from torchtext import data
from torchtext.legacy import data
from argparse import ArgumentParser
from embedding import EmbedVector
from sklearn.metrics.pairwise import euclidean_distances

parser = ArgumentParser(description="Training predicate vector learning")
parser.add_argument('--qa_mode', type=str, required=True, help='options are GRU, LSTM')
parser.add_argument('--embed_dim', type=int, default=250)
parser.add_argument('--no_cuda', action='store_false', help='do not use cuda', dest='cuda')
parser.add_argument('--gpu', type=int, default=0)  # Use -1 for CPU
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--seed', type=int, default=3435)
parser.add_argument('--dev_every', type=int, default=10000)
# parser.add_argument('--log_every', type=int, default=2000)
parser.add_argument('--log_every', type=int, default=200)
parser.add_argument('--patience', type=int, default=12)
parser.add_argument('--best_prefix', type=str, default='pred')
parser.add_argument('--output_channel', type=int, default=300)
parser.add_argument('--num_layer', type=int, default=2)
parser.add_argument('--rnn_fc_dropout', type=float, default=0.3)
parser.add_argument('--hidden_size', type=int, default=300)
parser.add_argument('--rnn_dropout', type=float, default=0.3)
parser.add_argument('--clip_gradient', type=float, default=0.6, help='gradient clipping')
parser.add_argument('--vector_cache', type=str, default="data/sq_glove300d.pt")
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--fix_embed', action='store_false', dest='train_embed')
parser.add_argument('--output', type=str, default='preprocess')
# parser.add_argument('--output', type=str, default='preprocess2')
args = parser.parse_args()


################## Prepare training and validation datasets ##################
pre_dic = {}  # Dictionary for predicates
for line in open(os.path.join('preprocess/', 'relation2id.txt'), 'r', encoding='utf-8'):
    items = line.strip().split("|")
    pre_dic[items[0]] = int(items[1])

# Embedding for predicates
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
predicates_emb = torch.from_numpy(predicates_emb)


# Set up the data for training
outfile = open(os.path.join('preprocess/', 'pred_train.txt'), 'w', encoding='utf-8')
for line in open(os.path.join('preprocess/', 'formatted_question.txt'), 'r', encoding='utf-8'):
    items = line.strip().split("\t")
    if items[1] in pre_dic:
        outfile.write("{}\t{}\n".format(items[3], pre_dic[items[1]]))   # 存储 问题 和 pred idx
outfile.close()
# for filename in ['train.txt', 'valid.txt']:
#     outfile = open(os.path.join(args.output, 'pred_' + filename), 'w', encoding='utf-8')
#     for line in open(os.path.join(args.output, filename), 'r', encoding='utf-8'):
#         items = line.strip().split("\t")
#         if items[3] in pre_dic:
#             outfile.write("{}\t{}\n".format(items[5], pre_dic[items[3]]))  # 存储 问题 和 pred idx
#             # pred_list.append(entities_emb[mid_dic[items[4]], :] - entities_emb[mid_dic[items[1]], :])
#             # token = list(compress(items[5].split(), [element == 'O' for element in items[6].split()]))
#             # if not token:
#     outfile.close()

# synthetic_flag = True
# # 补充问题
# if synthetic_flag:
#     names_map = {}  # 储存所有 entity 的 id 和 实体词
#     for i, line in enumerate(open(os.path.join(args.output, 'names.trimmed.txt'), 'r', encoding='utf-8')):
#         items = line.strip().split("\t")
#         if len(items) != 2:
#             print("ERROR: line - {}".format(line))
#             continue
#         entity = items[0]
#         literal = items[1].strip()
#         if literal != "" and (names_map.get(entity) is None or len(names_map[entity].split()) > len(literal.split())):
#             names_map[entity] = literal
#     seen_fact = []  # 储存 (headid, pred 实体词)
#     for line in open(os.path.join(args.output, 'train.txt'), 'r', encoding='utf-8'):
#         items = line.strip().split("\t")
#         names_map[items[1]] = items[2]
#         seen_fact.append((items[1], items[3]))
#     seen_fact = set(seen_fact)
#     whereset = {'location', 'place', 'geographic', 'region', 'places'}
#     whoset = {'composer', 'people', 'artist', 'author', 'publisher', 'directed', 'developer', 'director', 'lyricist',
#               'edited', 'parents', 'instrumentalists', 'produced', 'manufacturer', 'written', 'designers', 'producer'}
#     outfile = open(os.path.join(args.output, 'pred_train.txt'), 'a', encoding='utf-8')
#     for line in open(os.path.join(args.output, 'transE_valid.txt'), 'r', encoding='utf-8'):
#         items = line.strip().split("\t")
#         if (items[0], items[2]) not in seen_fact and names_map.get(items[0]) is not None:  # 找出 tail
#             name = names_map[items[0]]
#             tokens = items[2].replace('.', ' ').replace('_', ' ').split()
#             seen = set()
#             clean_token = [token for token in tokens if not (token in seen or seen.add(token))]
#             question = 'what is the ' + ' '.join(clean_token) + ' of ' + name
#             for token in clean_token:
#                 if token in whereset:
#                     question = 'where is ' + ' '.join(clean_token) + ' of ' + name
#                     break
#                 elif token in whoset:
#                     question = 'who is the ' + ' '.join(clean_token) + ' of ' + name
#                     break
#             outfile.write("{}\t{}\n".format(question, pre_dic[items[2]]))
#     outfile.close()
#     del names_map, pre_dic

################## Set random seed for reproducibility ##################
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True

if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("Warning: You have Cuda but not use it. You are using CPU for training.")

# Dictionary and embedding for words
if os.path.isfile(args.vector_cache):
    stoi, vectors, words_dim = torch.load(args.vector_cache)
else:
    print("Error: Need word embedding pt file")
    exit(1)


################## Load the datasets ##################
TEXT = data.Field(lower=True)
ED = data.Field(sequential=False, use_vocab=False)
train, dev = data.TabularDataset.splits(path='preprocess/', train='pred_train.txt', validation='pred_train.txt', format='tsv', fields=[('text', TEXT), ('mid', ED)])
test = data.TabularDataset(path=os.path.join('preprocess/', 'formatted_question.txt'), format='tsv', fields=[('entity', None), ('relation', None), ('obj', None), ('text', TEXT), ('ed', None)])


freebase_dataset_train = data.TabularDataset(path=os.path.join('freebase/', 'dete_train_org.txt'), format='tsv', fields=[('text', TEXT), ('ed', None)])
freebase_field = [('id', None), ('sub', None), ('entity', None), ('relation', None), ('obj', None), ('text', TEXT), ('ed', None)]
freebase_dataset_dev, freebase_dataset_test = data.TabularDataset.splits(path='freebase/', validation='valid.txt', test='test.txt', format='tsv', fields=freebase_field)

TEXT.build_vocab(train, dev, test, freebase_dataset_train, freebase_dataset_dev, freebase_dataset_test)

match_embedding = 0
TEXT.vocab.vectors = torch.Tensor(len(TEXT.vocab), words_dim)
for i, token in enumerate(TEXT.vocab.itos):
    wv_index = stoi.get(token, None)
    if wv_index is not None:
        TEXT.vocab.vectors[i] = vectors[wv_index]
        match_embedding += 1
    else:
        TEXT.vocab.vectors[i] = torch.FloatTensor(words_dim).uniform_(-0.25, 0.25)
print("Word embedding match number {} out of {}".format(match_embedding, len(TEXT.vocab)))

del stoi, vectors

if args.cuda:
    train_iter = data.Iterator(train, batch_size=args.batch_size, device=torch.device('cuda', args.gpu), train=True,
                               repeat=False, sort=False, shuffle=True, sort_within_batch=False)
    dev_iter = data.Iterator(dev, batch_size=args.batch_size, device=torch.device('cuda', args.gpu), train=False,
                             repeat=False, sort=False, shuffle=False, sort_within_batch=False)
else:
    train_iter = data.Iterator(train, batch_size=args.batch_size, train=True, repeat=False, sort=False, shuffle=True,
                               sort_within_batch=False)
    dev_iter = data.Iterator(dev, batch_size=args.batch_size, train=False, repeat=False, sort=False, shuffle=False,
                             sort_within_batch=False)

config = args
config.words_num = len(TEXT.vocab)
config.label = args.embed_dim
config.words_dim = words_dim
model = EmbedVector(config)

model.embed.weight.data.copy_(TEXT.vocab.vectors)
if args.cuda:
    modle = model.to(torch.device("cuda:{}".format(args.gpu)))
    print("Shift model to GPU")
    # Embedding for MID
    predicates_emb = predicates_emb.cuda()

total_num = len(dev)
print(config)
print("VOCAB num", len(TEXT.vocab))
print("Train instance", len(train))
print("Dev instance", total_num)
print(model)

parameter = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.MSELoss()

early_stop = False
best_accu = 0
best_loss = total_num
iterations = 0
iters_not_improved = 0
num_dev_in_epoch = (len(train) // args.batch_size // args.dev_every) + 1
patience = args.patience * num_dev_in_epoch  # for early stopping
epoch = 0
start = time.time()
print('  Time Epoch Iteration Progress    (%Epoch)   Loss')
log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f}'.split(','))

while True:
    if early_stop:
        print("Early Stopping. Epoch: {}, Best Dev accuracy: {}, loss: {},".format(epoch, best_accu, best_loss))
        break
    epoch += 1

    # 不让模型训练太久
    if epoch == 12:
        break

    train_iter.init_epoch()
    for batch_idx, batch in enumerate(train_iter):
        # Batch size : (Sentence Length, Batch_size)
        iterations += 1
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(batch), predicates_emb[batch.mid, :])
        loss.backward()
        # clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()

        # evaluate performance on validation set periodically
        # if iterations % args.dev_every == 0:
        if iterations % 500 == 0:
            model.eval()
            dev_iter.init_epoch()
            n_dev_correct = 0
            dev_loss = 0
            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                batch_size = dev_batch.text.size()[1]
                answer = model(dev_batch)

                # can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
                # learned_pred = euclidean_distances(answer.cpu().data.numpy(), predicates_emb).argmin(axis=1)
                learned_pred = euclidean_distances(answer.cpu().data.numpy(), predicates_emb.cpu().numpy()).argmin(axis=1)

                n_dev_correct += sum(dev_batch.mid.cpu().data.numpy() == learned_pred)
                dev_loss += criterion(answer, predicates_emb[dev_batch.mid, :]).item() * batch_size

            curr_accu = n_dev_correct / total_num
            total_loss = dev_loss/total_num
            print('Dev loss: {}, accuracy: {}'.format(total_loss, curr_accu))

            # update model
            if curr_accu > best_accu:  # total_loss < best_model
                best_accu = curr_accu
                best_loss = total_loss
                iters_not_improved = 0
                # save model, delete previous 'best_snapshot' files
                torch.save(model, os.path.join(args.output, args.best_prefix + '_best_model.pt'))
            else:
                iters_not_improved += 1
                if iters_not_improved > patience:
                    early_stop = True
                    break

        if iterations % args.log_every == 1:
            print(log_template.format(time.time() - start,
                                          epoch, iterations, 1 + batch_idx, len(train_iter),
                                          100. * (1 + batch_idx) / len(train_iter), loss.item(), ' ' * 8, ' ' * 12))
