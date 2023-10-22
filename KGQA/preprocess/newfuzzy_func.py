import math
from collections import Counter
from fuzzywuzzy import fuzz
import nltk


def get_ent_from_fuzzy(input_word):
    # count[word]可以得到每个单词的词频， sum(count.values())得到整个句子的单词总数
    def tf(word, count):
        return count[word] / sum(count.values())

    # 统计的是含有该单词的句子数
    def n_containing(word, count_list):
        return sum(1 for count in count_list if word in count)

    # len(count_list)是指句子的总数，n_containing(word, count_list)是指含有该单词的句子的总数，加1是为了防止分母为0
    def idf(word, count_list):
        return math.log(len(count_list) / (1 + n_containing(word, count_list)))

    def computation(string1, string2):
        score = fuzz.token_sort_ratio(string1, string2)
        return score

    entities = set()
    word_list = []
    stop_words = [" - ", "the", "a", "an"]

    with open("train.txt", "r", encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            arr = line.split("|")
            if arr[0] not in entities:
                tokens = nltk.word_tokenize(arr[0].lower().strip())
                for word in stop_words:
                    if word in tokens:
                        tokens.remove(word)

                if " " in tokens:
                    tokens.remove(" ")

                entities.add(" ".join(tokens))
                word_list.append(tokens)

    with open("simplified_form.txt", "r", encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if line != "\n":
                arr = line.split("\t")
                if arr[0] != " ":
                    if arr[0].strip() not in entities:
                        tokens = nltk.word_tokenize(arr[0].lower().strip())
                        for word in stop_words:
                            if word in tokens:
                                tokens.remove(word)

                        if " " in tokens:
                            tokens.remove(" ")

                        entities.add(" ".join(tokens))
                        word_list.append(tokens)

                    if arr[1].strip() not in entities:
                        tokens = nltk.word_tokenize(arr[1].lower().strip())
                        for word in stop_words:
                            if word in tokens:
                                tokens.remove(word)

                        if " " in tokens:
                            tokens.remove(" ")

                        entities.add(" ".join(tokens))
                        word_list.append(tokens)

    # countlist：每个单词的tf的集合
    # word_list：所有entity包含单词的集合（去除重复的单词）
    countlist = []
    for i in range(len(word_list)):
        count = Counter(word_list[i])
        countlist.append(count)

    word_list = set(sum(word_list, []))

    """
    通过idf，得出输入entity中idf最高的单词，用这个单词通过fuzzy获得word_list中的多个candidate word
    通过candidate words过滤掉干扰的entity，获得candidate entities
    """
    target_word = input_word
    target_tokens = nltk.word_tokenize(target_word)
    for word in stop_words:
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
        idf_score = idf(word, countlist)
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
        for word in word_list:
            word_candidates[index] = sorted(word_candidates[index], key=lambda k: k['score'], reverse=True)
            score = fuzz.ratio(word, target_token['word'])
            if score > word_candidates[index][-1]["score"]:
                word_candidates[index].pop()
                word_candidates[index].append({"string": word, "score": score})

    # 二维list变一维
    word_candidates = [j for i in word_candidates for j in i]

    # 利用candidate words获取candidate entities
    candidate_entities = []
    for entity in entities:
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
        score = computation(string1, target_word)
        if score > scores[len(scores) - 1]['score']:
            scores.pop()
            scores.append({"string": string1, "score": score})

    """
    score最高的是最后答案；如果有相同的score，则选择那个长度短的entity作为最后的答案
    """
    scores = sorted(scores, key=lambda k: k['score'], reverse=True)
    final_result = scores[0]['string']
    max_score = scores[0]['score']
    return final_result + '   ' + str(max_score)  # 可能得到的是缩写, 所以还要将缩写转换为一般entity

print(get_ent_from_fuzzy("compu"))


