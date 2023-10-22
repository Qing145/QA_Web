from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from collections import Counter
import nltk
import math
import sys
import os


class EntityLinking:
    def __init__(self):
        """
        类属性
        """
        # 别名列表 {"it":"information technology"}
        self.alias = {}
        # 单词列表
        self.word_list = []
        # 去除单个字母的stop words，因为有单个字母的需求，比如Core A，不能把A过滤了
        self.stop_words = [word for word in set(stopwords.words('english')) if len(word) > 1]

        """
        处理别名列表
        """
        with open('/'.join(os.path.abspath(__file__).split('/')[:-1]) + '/preprocess/alias.txt', 'r', encoding='utf-8') as file:
            for line in file.readlines():
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

                        # 将tokens加入到word list中，为了计算tf-idf
                        self.word_list.append(tokens)

                        # 组合处理之后的tokens
                        alias_name = " ".join(tokens).lower()
                        if alias_name in self.alias.keys():
                            # 如果别名键已经存在，则直接追加
                            self.alias[alias_name].append(arr[0])
                            # # 检查是否已经存过此 key-value 对
                            # if arr[0] not in self.alias[alias_name]:
                            #     print(alias_name)
                            #     self.alias[alias_name].append(arr[0])
                        else:
                            # 如果别名键已经存在，则直接创建
                            self.alias[alias_name] = [arr[0]]

                # 把自身加进去alias
                if arr[0].lower() not in self.alias.keys():
                    # 把stop words都处理了
                    tokens = nltk.word_tokenize(arr[0].lower().strip())
                    for word in self.stop_words:
                        if word in tokens:
                            tokens.remove(word)
                    if " " in tokens:
                        tokens.remove(" ")
                    alias_name = " ".join(tokens).lower()

                    # 加入alias
                    self.alias[alias_name] = [arr[0]]
                    # 加入word list
                    self.word_list.append(tokens)
                else:
                    # 把stop words都处理了
                    tokens = nltk.word_tokenize(arr[0].lower().strip())
                    for word in self.stop_words:
                        if word in tokens:
                            tokens.remove(word)
                    if " " in tokens:
                        tokens.remove(" ")
                    alias_name = " ".join(tokens).lower()

                    self.alias[alias_name].append(arr[0])

        # 去重
        for item in self.alias.keys():
            self.alias[item] = list(set(self.alias[item]))

        """
        countlist：每个单词的tf的集合
        word_list：所有entity包含单词的集合（去除重复的单词）
        """
        self.countlist = []
        for i in range(len(self.word_list)):
            count = Counter(self.word_list[i])
            self.countlist.append(count)

        self.word_list = set(sum(self.word_list, []))

    """
    tf-idf相关函数
    """

    # count[word]可以得到每个单词的词频， sum(count.values())得到整个句子的单词总数
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

    # 将tf和idf相乘
    def tfidf(self, word, count, count_list):
        return self.tf(word, count) * self.idf(word, count_list)

    # 通过tf-idf过滤掉一些alias，为了提高速度
    def get_candidate_alias(self, target_word):
        """
        通过idf，得出输入entity中idf最高的单词，用这个单词通过fuzzy获得word_list中的多个candidate word
        通过candidate words过滤掉干扰的alias，获得candidate alias
        """
        # 处理stop words和多个空格
        target_tokens = nltk.word_tokenize(target_word)
        for word in self.stop_words:
            if word in target_tokens:
                target_tokens.remove(word)
        if " " in target_tokens:
            target_tokens.remove(" ")

        # 根据idf，保留top-n target tokens
        if len(target_tokens) > 1:
            target_token_top_n = 2
        else:
            target_token_top_n = 1

        count = Counter(target_tokens)
        target_dics = [{"word": word, "tf_idf_score": self.tfidf(word, count, self.countlist)} for word in count]
        target_dics = sorted(target_dics, key=lambda k: k['tf_idf_score'], reverse=True)

        # 保留top-n candidate words
        top_n = 2
        word_candidates = []
        for i in range(target_token_top_n):
            word_candidates.append([{"string": "", "score": 0}] * top_n)
        for index, target_token in enumerate(target_dics[:target_token_top_n]):
            for word in self.word_list:
                word_candidates[index] = sorted(word_candidates[index], key=lambda k: k['score'], reverse=True)
                if word[0:1] == target_token['word'][0:1]:
                    score = fuzz.ratio(word, target_token['word'])

                    if score > word_candidates[index][-1]["score"]:
                        word_candidates[index].pop()
                        word_candidates[index].append({"string": word, "score": score})
            if max(word_candidates[index], key=lambda k: k['score'])['score'] == 100:
                word_candidates[index] = [max(word_candidates[index], key=lambda k: k['score'])]

        # 二维list变一维
        word_candidates = [j for i in word_candidates for j in i]
        # print(word_candidates)


        # 利用candidate words获取candidate alias
        candidate_alias = []
        for item in self.alias.keys():
            arr = nltk.word_tokenize(item)
            for i in range(len(word_candidates)):
                if word_candidates[i]['string'] in arr:
                    candidate_alias.append(item)
                    break
        candidate_alias = set(candidate_alias)  # 去重

        return candidate_alias

    # 计算两个实体的fuzzy score
    def computation(self, alias, detected_entity_name):
        # alias_tokens = nltk.word_tokenize(alias.lower().strip())
        # detected_entity_name_tokens = nltk.word_tokenize(detected_entity_name.lower().strip())
        #
        # # 算出每个token对应的best score，加起来除以tokens的个数
        # total_score = 0
        # for token1 in detected_entity_name_tokens:
        #     best_score = 0
        #     for token2 in alias_tokens:
        #         # fuzzy原理：Levensthein编辑距离 - (编辑次数) / (两个字符串的字母数和)
        #         score = fuzz.ratio(token1, token2)
        #         if score > best_score:
        #             best_score = score
        #     total_score += best_score
        #
        # final_score = total_score / len(detected_entity_name_tokens)
        # return final_score

        string1 = alias.lower().strip()
        string2 = detected_entity_name.lower().strip()
        final_score = (fuzz.token_sort_ratio(string1, string2) + fuzz.token_set_ratio(string1, string2)) / 2
        return final_score

    # 获取候选实体
    def get_candidate_entites(self, detected_entity_name, candidate_alias):
        """
        在candidate entities中利用fuzzy获得top-n entities，此处是利用整个句子下去匹配
        """
        top_n = 5
        scores = [{"string": "", "score": 0}] * top_n

        for alias in candidate_alias:
            scores = sorted(scores, key=lambda k: k['score'], reverse=True)
            score = self.computation(alias, detected_entity_name)

            if score > scores[len(scores) - 1]['score']:
                scores.pop()
                scores.append({"string": alias, "score": score})

        # scores = sorted(scores, key=lambda k: k['score'], reverse=True)
        # 根据阈值，获得最终的候选实体集
        candidate_entities = []
        for score in scores:
            if score['score'] > 75:
                candidate_entities.extend(self.alias[score['string']])
                print(score)
        candidate_entities = list(set(candidate_entities))
        return candidate_entities

    # 实体连接主函数
    def entity_link(self, detected_entity):
        # 把stop words都处理了
        tokens = nltk.word_tokenize(detected_entity.lower().strip())
        for word in self.stop_words:
            if word in tokens:
                tokens.remove(word)
        if " " in tokens:
            tokens.remove(" ")
        detected_entity_name = " ".join(tokens).lower()

        # 如果存在一样的则直接返回entity list，返回的是一个数组
        if detected_entity_name in self.alias.keys():
            return self.alias[detected_entity_name]

        # 如果没有，则从单个token层面，利用tf-idf过滤掉一些alias，附带fuzzy纠错
        detected_entity_alias = self.get_candidate_alias(detected_entity_name)

        # 获取候选实体集
        candidate_entities = self.get_candidate_entites(detected_entity_name, detected_entity_alias)

        return candidate_entities


if __name__ == '__main__':
    while True:
        f = EntityLinking()  # 使用自己编写的 fuzzy 方法
        word = input('plz input: ')
        res = f.entity_link(word)  # 可能得到的是缩写, 所以还要将缩写转换为一般entity
        if res:  # 如果找得出大于阈值的答案
            print(res)
        else:
            print('NOT EXISTS!!!')
