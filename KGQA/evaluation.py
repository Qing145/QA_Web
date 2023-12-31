from collections import Counter


def get_span(label, index2tag, type):
    '''
    :param label: [2 2 2 2 2 3 3 3 2 1 1 1 1]
    :param index2tag: ['<unk>' '<pad>' 'O' 'I']
    :param type: type = false
    :return: 返回 'I' 的 所有位置
    '''
    span = []
    st = -1  # start
    en = -1  # end
    flag = False  # 标记前面是否有 'I'
    tag = []
    for k in range(len(label)):
        if index2tag[label[k]][0] == 'I' and flag == False:
            flag = True
            st = k
            if type:
                tag.append(index2tag[label[k]][2:])
        if index2tag[label[k]][0] == 'I' and flag == True:
            if type:
                tag.append(index2tag[label[k]][2:])
        if index2tag[label[k]][0] != 'I' and flag == True:  # 一个 head 遍历结束
            flag = False
            en = k
            if type:
                max_tag_counter = Counter(tag)
                max_tag = max_tag_counter.most_common()[0][0]
                span.append((st, en, max_tag))
            else:
                span.append((st, en))
            st = -1
            en = -1
            tag = []
    if st != -1 and en == -1:
        en = len(label)
        if type:
            max_tag_counter = Counter(tag)
            max_tag = max_tag_counter.most_common()[0][0]
            span.append((st, en, max_tag))
        else:
            span.append((st, en))

    return span


def evaluation(gold, pred, index2tag, type):
    right = 0
    predicted = 0
    total_en = 0
    #fout = open('log.valid', 'w')
    for i in range(len(gold)):
        gold_batch = gold[i]
        pred_batch = pred[i]

        for j in range(len(gold_batch)):
            gold_label = gold_batch[j]
            pred_label = pred_batch[j]
            gold_span = get_span(gold_label, index2tag, type)
            pred_span = get_span(pred_label, index2tag, type)
            #fout.write('{}\t{}\n'.format(gold_span, pred_span))
            total_en += len(gold_span)
            predicted += len(pred_span)
            for item in pred_span:
                if item in gold_span:
                    right += 1
    if predicted == 0:
        precision = 0
    else:
        precision = right / predicted
    if total_en == 0:
        recall = 0
    else:
        recall = right / total_en
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    #fout.flush()
    #fout.close()
    return precision, recall, f1


def get_names_for_entities(namespath):
    print("getting names map...")
    names = {}
    with open(namespath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            items = line.strip().split("\t")
            if len(items) != 2:
                print("ERROR: line - {}".format(line))
                continue
            entity = items[0]
            literal = items[1].strip()
            if literal != "":
                if names.get(literal) is None:
                    names[literal] = [(entity)]
                else:
                    names[literal].append(entity)
                    #print('ERROR: Entities with the same name!')
    return names