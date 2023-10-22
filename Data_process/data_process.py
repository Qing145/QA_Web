# JiangHao
import os

path = './data'  # 文件夹目录
files = os.listdir(path)  # 得到文件夹下的所有文件名称

txts = []
entitys = []
relations = []

for file in files:  # 遍历文件夹
    position = path + '/' + file  # 构造绝对路径，"\\"，其中一个'\'为转义符
    with open(position, "r", encoding='utf-8') as f:  # 打开文件
        for line in f.readlines():
            line = line.strip("\n").strip().replace("  ", " ")
            arr = line.split("|")
            arr[0] = arr[0].strip()
            arr[1] = arr[1].strip()
            arr[2] = arr[2].strip()

            if arr[0] not in entitys:
                entitys.append(arr[0])
            if arr[1] not in relations:
                relations.append(arr[1])
            if arr[2] not in entitys:
                entitys.append(arr[2])

            txts.append(arr)

with open('./entity2id.txt', 'a') as file:
    file.truncate(0)
    file.write(str(len(entitys)) + "\n")
    for index, entity in enumerate(entitys):
        file.write(str(entity) + "\t" + str(index) + "\n")

with open('./relation2id.txt', 'a') as file:
    file.truncate(0)
    file.write(str(len(relations)) + "\n")
    for index, relaton in enumerate(relations):
        file.write(str(relaton) + "\t" + str(index) + "\n")

with open('./train2id.txt','a') as file:
    file.truncate(0)
    file.write(str(len(txts)) + "\n")
    for index, arr in enumerate(txts):
        string = str(entitys.index(arr[0])) + "\t" + str(entitys.index(arr[2])) + "\t" + str(relations.index(arr[1])) + "\n"
        file.write(string)
