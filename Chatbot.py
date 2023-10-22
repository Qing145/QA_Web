# JiangHao
# coding

import os
from Rule_based_QA.RuleBasedQA import RuleBasedQA
from Predefined_QA.Predefined_QA import Predefined_QA
from KGQA.KGQA import KGQA
from fuzzywuzzy import fuzz
from Content_based_QA.Content_based_QA import Conten_based_QA


class ChatBot:
    def __init__(self):
        """************无法回答返回模板************"""
        self.no_answer = 'Sorry, I may not be able to answer your question!'

        # 加载基于规则的问答模块
        self.rule_based_QA_handler = RuleBasedQA()
        # 加载预定义问答模块
        self.predefined_QA_handler = Predefined_QA()
        # 加载基于内容QA模块
        self.conten_based_QA = Conten_based_QA()
        # 加载基于知识库的QA模块
        self.KGQA_handler = KGQA()

        # 存放Triples的字典，key：relation，value: [(head, tail),(head, tail)..]
        self.triples = {}

        """************加载所有triples************"""
        # 当前路径
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])

        with open('./KGQA/preprocess/train.txt') as file:
            for line in file.readlines():
                if line != "\n" and line != "":
                    arr = line.split("|")
                    arr[0] = arr[0].strip("\n").strip(" ").lower()
                    arr[1] = arr[1].strip("\n").strip(" ").lower()
                    arr[2] = arr[2].strip("\n").strip(" ")

                    if self.triples.__contains__(arr[1]):
                        self.triples[arr[1]].append((arr[0], arr[2]))
                    else:
                        self.triples[arr[1]] = []
                        self.triples[arr[1]].append((arr[0], arr[2]))

    # 获得预定义模板回复
    def get_rule_reply(self, question):
        reply = self.predefined_QA_handler.get_rule_reply(question)
        if reply[2] != -1:
            relation = self.predefined_QA_handler.type_to_relation(reply[2])
            entity_arr = []
            if self.triples.__contains__(relation):
                for triple in self.triples[relation]:
                    entity_arr.append(triple[0])

            return {"reply": reply, "related_entity": entity_arr}
        else:
            return {"reply": reply, "related_entity": []}

    # 获得预定义模板答案
    def get_rule_answer(self, entity, type):
        relation = self.predefined_QA_handler.type_to_relation(type)
        best_score = 0
        if self.triples.__contains__(relation):
            for triple in self.triples[relation]:
                score = fuzz.token_sort_ratio(triple[0], entity)
                if score > best_score:
                    answer = triple[1]
                    best_score = score

            if best_score < 60:
                answer = self.no_answer
        else:
            answer = self.no_answer

        return answer

    # 获取规则模型答案
    def get_rule_based_answer(self, question):
        answer = self.rule_based_QA_handler.get_answer(question)

        if answer == "":
            final_answer = ""
        else:
            final_answer = answer

        return final_answer

    # 获取答案主函数
    def get_answer(self, question):
        rule_based_answer = self.get_rule_based_answer(question)

        # 匹配到规则，但是没匹配到答案
        # if rule_based_answer == "":
        #     answer, no_answer = self.conten_based_QA.get_answer(question)
        #     if no_answer > 0:
        #         return answer
        #     else:
        #         return self.no_answer

        # 匹配不到规则，则进入KGQA
        if rule_based_answer == "No Rule Match" or rule_based_answer == "":
            head_entity = self.KGQA_handler.get_head_entity(question)
            KGQA_answer = self.KGQA_handler.get_answer()
            if KGQA_answer[0] == "KGQA":
                return KGQA_answer[1]
            elif KGQA_answer[0] == "mix-mode":
                answer, no_answer = self.conten_based_QA.get_answer(question)
                if no_answer > 0:
                    final_answer = "Trigger ambiguity mechanism\n" + "* Possible Answer 1\n" + \
                                   KGQA_answer[
                                       1] + "\n\n" + "* Possible Answer 2\n" + answer
                    return final_answer
                else:
                    return "Trigger ambiguity mechanism\n" + "* Possible Answer\n" + \
                           KGQA_answer[1]
            else:
                answer, no_answer = self.conten_based_QA.get_answer(question)
                if no_answer > 0:
                    return answer
                else:
                    return self.no_answer
        else:
            return rule_based_answer


if __name__ == '__main__':
    chatBot = ChatBot()
    # print(chatBot.get_rule_answer("Faculty of Applied Science and Textiles", 0))
    # print(chatBot.get_rule_based_answer("where is core A"))

    result = chatBot.get_answer("who is current president of polyu")
    print(result)
