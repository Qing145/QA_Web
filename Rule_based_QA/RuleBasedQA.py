# coding: utf-8
# File: question_classifier.py
import json
import os
import ahocorasick
import re
import string
import nltk
from nltk.corpus import stopwords
import spacy
from fuzzywuzzy import fuzz


class RuleBasedQA:
    def __init__(self):
        self.cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        # 去除单个字母的stop words，因为有单个字母的需求，比如Core A，不能把A过滤了
        self.stop_words = list(set(stopwords.words('english')))
        self.stop_words.extend(['many', 'much', "'s"])
        self.alias = {}
        self.triples = []
        self.nlp = spacy.load("en_core_web_sm")
        self.relation = ""
        self.entity = ""

        # 首次匹配问题类型规则加载
        with open(os.path.join(self.cur_dir,
                               'data/question_first_classify.json')) as file:
            json_file = json.load(file)
            self.first_classify_rule_arrs = json_file["rules_array"]

        # 数量问题类型规则加载
        with open(os.path.join(self.cur_dir,
                               'data/question_number_classify.json')) as file:
            json_file = json.load(file)
            self.number_classify_rules_arrs = json_file["rules_array"]

        # 时间问题类型规则加载
        with open(os.path.join(self.cur_dir,
                               'data/question_time_classify.json')) as file:
            json_file = json.load(file)
            self.time_classify_rules_arrs = json_file["rules_array"]

        # 金钱问题类型规则加载
        with open(os.path.join(self.cur_dir,
                               'data/question_tuition_classify.json')) as file:
            json_file = json.load(file)
            self.entity_tuition_rules_arrs = json_file["rules_array"]

        """
        处理别名列表
        """
        with open(self.cur_dir + '/data/alias.txt', 'r',
                  encoding='utf-8') as file:
            stop_words = [word for word in self.stop_words if len(word) > 1]
            stop_words.remove("it")
            for line in file.readlines():
                arr = line.strip("\n").split("|")

                # 有别名
                if arr[1] != "":
                    # 别名列表
                    alias_list = arr[1].split(",")
                    # 将所有别名以键值对加进alias
                    for alias_item in alias_list:
                        # 把别名里的stop words都处理了
                        tokens = nltk.word_tokenize(alias_item.lower().strip())
                        for word in stop_words:
                            if word in tokens:
                                tokens.remove(word)
                        if " " in tokens:
                            tokens.remove(" ")

                        # 组合处理之后的tokens
                        alias_name = " ".join(tokens).lower()

                        if "( " in alias_name:
                            alias_name = alias_name.replace("( ", "(")
                        if " )" in alias_name:
                            alias_name = alias_name.replace(" )", ")")

                        if alias_name in self.alias.keys():
                            # 如果别名键已经存在，则直接追加
                            self.alias[alias_name].append(arr[0].lower())
                        else:
                            # 如果别名键不存在，则直接创建
                            self.alias[alias_name] = [arr[0].lower()]

                # 把自身的stop words都处理了
                tokens = nltk.word_tokenize(arr[0].lower().strip())
                for word in stop_words:
                    if word in tokens:
                        tokens.remove(word)
                if " " in tokens:
                    tokens.remove(" ")
                alias_name = " ".join(tokens).lower()

                if "( " in alias_name:
                    alias_name = alias_name.replace("( ","(")
                if " )" in alias_name:
                    alias_name = alias_name.replace(" )", ")")

                # 把自身加进去alias
                if alias_name not in self.alias.keys():
                    # 加入alias
                    self.alias[alias_name] = [arr[0].lower()]
                else:
                    self.alias[alias_name].append(arr[0].lower())

        # 去重
        for item in self.alias.keys():
            self.alias[item] = list(set(self.alias[item]))

        with open(self.cur_dir + '/data/train.txt', 'r',
                  encoding='utf-8') as file:
            for line in file.readlines():
                arr = line.strip("\n").split("|")
                arr[0] = arr[0].lower()
                self.triples.append(arr)

        print('Rule-based model init finished ......')

        return

    # 规则匹配 返回relation
    def rule_match(self, string):
        # 把问题化成小写
        string = string.lower()

        # 首次匹配问题类型
        type = ""
        for rule_obj in self.first_classify_rule_arrs:
            is_break = False
            for rule in rule_obj["rules"]:
                if re.match(r"" + rule, string):
                    type = rule_obj["type"]
                    is_break = True
                    break

            if is_break:
                break

        # 根据问题类型进行二次匹配，提取relation
        question_type = {
            "type": "",
            "level": 0
        }

        # 根据首次匹配类型进行再次细分匹配
        if type == "Number":
            for rule_obj in self.number_classify_rules_arrs:
                for rule in rule_obj["rules"]:
                    if re.match(r"" + rule, string):
                        # 根据匹配等级进行匹配，等级（1-3），越低代表匹配精度越低
                        if rule_obj["level"] == 3:
                            question_type["type"] = rule_obj["type"]
                            question_type["level"] = rule_obj["level"]
                            break
                        elif rule_obj["level"] > question_type["level"]:
                            question_type["type"] = rule_obj["type"]
                            question_type["level"] = rule_obj["level"]
        elif type == "Time":
            for rule_obj in self.time_classify_rules_arrs:
                for rule in rule_obj["rules"]:
                    if re.match(r"" + rule, string):
                        # 根据匹配等级进行匹配，等级（1-3），越低代表匹配精度越低
                        if rule_obj["level"] == 3:
                            question_type["type"] = rule_obj["type"]
                            question_type["level"] = rule_obj["level"]
                            break
                        elif rule_obj["level"] > question_type["level"]:
                            question_type["type"] = rule_obj["type"]
                            question_type["level"] = rule_obj["level"]
        elif type == "Location":
            question_type["type"] = "Location"
            question_type["level"] = 3
        elif type == "Tuition":
            for rule_obj in self.entity_tuition_rules_arrs:
                for rule in rule_obj["rules"]:
                    if re.match(r"" + rule, string):
                        # 根据匹配等级进行匹配，等级（1-3），越低代表匹配精度越低
                        if rule_obj["level"] == 3:
                            question_type["type"] = rule_obj["type"]
                            question_type["level"] = rule_obj["level"]
                            break
                        elif rule_obj["level"] > question_type["level"]:
                            question_type["type"] = rule_obj["type"]
                            question_type["level"] = rule_obj["level"]

        return question_type["type"]

    def rule_entity_match(self, relation, question):
        # 根据relation找出所有候选entity
        entities_filtered_by_relation = [triple for triple in self.triples if
                                         triple[1].lower() == relation.lower()]
        candidate_entities = [triple[0] for triple in
                              entities_filtered_by_relation]

        for key in self.alias.keys():
            retB = list(
                set(self.alias[key]).intersection(set(candidate_entities)))
            if len(retB) != 0 and key not in candidate_entities:
                candidate_entities.append(key)

        if len(entities_filtered_by_relation) == 1:
            return entities_filtered_by_relation[0][0]

        # 将问题小写，然后去除所有动词
        question = question.lower()
        doc = self.nlp(question)
        question_tokens = [token.text for token in doc if token.tag_ != "VB"]
        question = " ".join(question_tokens)

        stop_words = self.stop_words
        # 根据relation添加停用词
        if relation == "Location":
            # 去除停用词中的单个单词，因为地点中有一些一个字母的entity关键词，比如Core A
            stop_words = [word for word in stop_words if len(word) > 1]
            stop_words.extend(['location'])
        elif relation == 'Elective Subjects':
            stop_words.extend(
                ['number', 'elective', 'selective', 'optional', 'courses',
                 'course', 'subject', 'subjects',
                 'class', 'classes'])
            # stop_words.remove('it')
        elif relation == 'Credits Required for Graduation':
            stop_words.extend(
                ['credits', 'credit', 'graduate', 'graduation', 'major',
                 'program', 'programme', 'required', 'require', 'took',
                 'needed'])
            if "it" in stop_words:
                stop_words.remove('it')
        elif relation == 'Compulsory Subjects':
            stop_words.extend(
                ['number', 'compulsory', 'required', 'optional', 'obligatory',
                 'course', "courses", 'subject', 'subjects',
                 'class', 'classes'])
            if "it" in stop_words:
                stop_words.remove('it')
        elif relation == 'Department':
            stop_words.extend(
                ['number', 'departments', 'department', 'sections', 'section',
                 'branch', 'branches'])
        elif relation == 'Graduates Number':
            stop_words.extend(
                ['number', 'student(s){0,1}', 'pupil', 'scholastic', 'graduate',
                 'graduation', 'finish school'])
        elif relation == 'Students Number':
            stop_words.extend(
                ['number', 'student(s){0,1}', 'pupil', 'scholastic'])
        elif relation == 'Faculty Members Number':
            stop_words.extend(
                ['number', 'faculty member(s){0,1}', 'teaching staff(s){0,1}',
                 'teaching and administrative staff(s){0,1}', 'staff(s){0,1}'])
        elif relation == 'Taught Programmes Number"':
            stop_words.extend(['taught programme(s){0,1}', 'major(s){0,1}',
                               'programme(s){0,1}', 'program(s){0,1}'])
        elif relation == 'Duration':
            stop_words.extend(
                ['duration', 'year', 'long', 'learning', 'study', 'learn',
                 'studying', 'learnt', 'studied', 'graduate',
                 'graduation', 'major', 'programme', 'program'])
        elif relation == 'Tuition':
            stop_words.extend(
                ['money', 'tuition fee', 'tuition', 'cost', 'expense', 'charge',
                 'expend', 'fee', 'pay for', 'graduate',
                 'graduation', 'major', 'programme', 'program'])

        for word in stop_words:
            pattern = '\\b(' + word + ')\\b'
            question = re.sub(pattern, '', question)

        question = re.sub(" +", " ", question)
        question = question.replace("?", "")
        question = question.strip()

        if question != "":
            final_entity = ""
            best_score = 0
            for candidate_entitie in candidate_entities:
                processed_question_tokens = self.nlp(question)
                score = fuzz.token_sort_ratio(question, candidate_entitie)
                if score > best_score:
                    final_entity = candidate_entitie
                    best_score = score

            if best_score > 60:               # print(final_entity)
                # print(self.alias.keys())

                return self.alias[final_entity][0]
            else:
                return ""
        else:
            return ""

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
            return "Credits required for graduation for " + t[0] + " is  " + t[
                2]
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
            return "There are the following Sports Facilities in " + t[
                0] + ":  " + t[2]
        elif pre == 'Student Halls of Residence':
            return "There are the following Student Halls of Residence in " + t[
                0] + ":  " + t[2]
        elif pre == 'Catering':
            return "There are the following Caterings in " + t[0] + ":  " + t[2]
        elif pre == 'Health Service':
            return "There are the following Health Services in " + t[
                0] + ":  " + t[2]
        elif pre == 'Global Student Hub':
            return "There are the following Global Student Hubs in " + t[
                0] + ":  " + t[2]
        elif pre == 'Student Lockers':
            return "There are the following Student Lockers in " + t[
                0] + ":  " + t[2]
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
            return "There are the following Taught Postgraduate Programmes in " + \
                   t[0] + ":  " + t[2]
        elif pre == 'Fund Type':
            return "The fund type of " + t[0] + " is  " + t[2]
        elif pre == 'Compulsory Subjects':
            return "The Compulsory Subjects in " + t[0] + " are:  " + t[2]
        elif pre == 'Elective Subjects':
            return "The Elective Subjects in " + t[0] + " are:  " + t[2]
        elif pre == 'Tuition':
            return "The tuition fee of " + t[0] + " is:  " + t[2]
        elif pre == 'Application':
            return "There are these real-world applications of " + t[
                0] + ":  " + t[2]
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

    def get_answer(self, question):
        relation = self.rule_match(question)

        # 没匹配到规则
        if relation == "":
            return "No Rule Match"

        # 匹配到规则，但是没匹配到entity
        entity = self.rule_entity_match(relation, question)
        if entity != "" and relation != "":
            self.relation = relation
            self.entity = entity
            if relation == "Elective Subjects" or relation == "Compulsory Subjects":
                tail = [triple[2] for triple in self.triples if
                        triple[0].lower() == entity.lower() and triple[
                            1].lower() == relation.lower()][0]
                subject_number = len(tail.split(", "))
                return "Number of " + relation + " for " + entity + " are " + str(
                    subject_number)
            else:
                tail = [triple[2] for triple in self.triples if
                        triple[0].lower() == entity.lower() and triple[
                            1].lower() == relation.lower()][0]
                final_answer = self.triple_to_answer([entity, relation, tail])
                return final_answer
        else:
            return ""


if __name__ == '__main__':
    handler = RuleBasedQA()
    questions = [
        "How many off-campus dormitories are there in PolyU"
    ]

    for question in questions:
        answer = handler.get_answer(question)
        if answer == "":
            print(question + "，无答案")
        else:
            print(question + "，答案为：" + str(answer))