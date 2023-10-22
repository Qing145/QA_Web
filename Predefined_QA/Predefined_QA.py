# JiangHao
import os


class Predefined_QA:
    def __init__(self):

        """************路径定义************"""
        # 当前路径
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        # 标签词路径
        tips_path = os.path.join(cur_dir, 'data/tips.txt')
        # FAQ路径
        faq_path = os.path.join(cur_dir, 'data/faq.txt')

        """************预定义数据加载************"""
        # 加载特征词
        self.tips_wds = []
        self.tips_reply = []
        self.tips_relation = []
        for item in open(tips_path):
            if item.strip():
                arr = item.split('|')
                self.tips_wds.append(arr[0])
                self.tips_reply.append(arr[1])
                self.tips_relation.append(arr[2])

        # 加载faq
        self.faq = []
        for item in open(faq_path):
            if item.strip():
                item = item.strip('\n')
                self.faq.append(item)

        print('Predefined QA model init finished ......')

    """************获取模板回复************"""

    def get_rule_reply(self, question):
        question = question.title()
        for index, item in enumerate(self.tips_wds):
            item = item.title()
            if question == item:
                return True, self.tips_reply[index], index
        return False, "", -1

    """************获取关系************"""

    def type_to_relation(self, index):
        return self.tips_relation[index].strip("\n").strip(" ").lower()


if __name__ == '__main__':
    Predefined_QA = Predefined_QA()
