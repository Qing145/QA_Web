import numpy as np
import os
import json

# entities_emb = np.fromfile(os.path.join('preprocess/', 'entity_250dim.txt'), dtype=np.float32)
# print(entities_emb)

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

print(len(arr))  # 562
print(len(arr[0]))  # 562
print(arr[0])

entities_emb = np.array(arr)
print(len(entities_emb))
print(len(entities_emb[0]))