import os
import json
import numpy as np

'''
计算尾实体之间的平均距离

KGQA中的最后阈值可取 尾实体之间的平均距离 的 1/n
'''


# 获取 entity:entityidx
mid_dic = {}  # Dictionary for MID     entity:entityidx
for line in open(os.path.join('entity2id.txt'), 'r', encoding='utf-8'):
    items = line.strip().split("|")
    mid_dic[items[0]] = int(items[1])


# Embedding for MID  获取 emb向量
n = 0
for line in open(os.path.join('entity_250dim.txt'), 'r', encoding='utf-8'):
    items = line.strip().split("\t")
    if len(items) != 2:
        print(items)
        continue
    n += 1

arr = [[]for _ in range(n)]
for line in open(os.path.join('entity_250dim.txt'), 'r', encoding='utf-8'):
    items = line.strip().split("\t")
    if len(items) != 2:
        print(items)
        continue
    # 现在 items[1] 是一堆字符串, 需要处理
    embedding = json.loads(items[1])
    arr[int(items[0])] = embedding  # idx 对应原本 entity 的编号
    # arr.append(items[1])

# 获取所有 tail entity 准备计算所有 tail entity 之间的平均距离
tail_ent = []
count = set()  # 避免重复
for line in open(os.path.join('train.txt'), 'r', encoding='utf-8'):
    items = line.strip().split("|")
    if len(items) != 3:
        print(items)
        continue
    # 取出tail
    if items[2] in count:
        continue
    if items[2] not in mid_dic:
        print(items)
        continue
    tail_ent.append(arr[mid_dic[items[2]]])
    count.add(items[2])

n = len(tail_ent)
print('entity_num: ' + str(n))
count = 0
tail_emb = np.array(tail_ent, dtype=np.float32)

sum_dis = 0
for i in range(n - 1):
    for j in range(i + 1, n):
        sum_dis += np.sqrt(np.sum(np.power(tail_emb[i] - tail_emb[j], 2)))  # 计算 每两个 tail entity 之间的距离
        count += 1

print('sum_dis: ' + str(sum_dis))
res = sum_dis / count  # 平均距离
print('average_dis: ' + str(res))
