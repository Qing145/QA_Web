# -*- coding: UTF-8 -*-

from flask import Flask, request, jsonify, Response, render_template
from flask_cors import CORS
from Chatbot import ChatBot
import json
import traceback
import os
import sys
from gevent import pywsgi

# 创建flask类的实例对象
app = Flask(__name__, template_folder='./templates',
            static_folder='./templates', static_url_path='')
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app, supports_credentials=True)

cur = os.getcwd()
sys.path.append(cur + "/KGQA/")

# 问答机器人
chatbot = ChatBot()


@app.route('/')
def index():
    return render_template('index.html')


# 加载标签和FAQ
@app.route('/getTips', methods=['GET'])
def getTips():
    try:
        dic = {
            "tips": chatbot.predefined_QA_handler.tips_wds,
            "faq": chatbot.predefined_QA_handler.faq
        }
        return_data = json.dumps(dic)
        return Response(return_data, status=200)

    except Exception as e:
        return Response(status=500)


# 获取答案
@app.route('/getAnswer', methods=['POST'])
def getAnswer():
    try:
        post_data = request.get_json()
        question = post_data.get('question')
        question = question.strip('\n').strip(' ')
        question = question.lower()

        is_stay_rule = post_data.get('is_stay_rule')

        """处于模板回复等待阶段"""
        if is_stay_rule != "false":
            type = int(post_data.get('type'))
            answer = chatbot.get_rule_answer(question, type)

            dic = {
                "answer": answer,
                "type": "",
                "is_stay_rule": "false",
                "origin": 0
            }
            return_data = json.dumps(dic)

            return Response(return_data, status=200)

        """不处于模板回复等待阶段"""
        reply = chatbot.get_rule_reply(question)
        # 获取模板回复，如果存在匹配模板则返回模板回复
        # 匹配到模板关键词
        if reply["reply"][0]:

            dic = {
                "answer": reply["reply"][1],
                "origin": 3,
                "related_entity": reply["related_entity"],
                "is_stay_rule": "true",
                "type": reply["reply"][2],
            }

            return_data = json.dumps(dic)

            return Response(return_data, status=200)
        # 匹配不到模板关键词
        else:
            # 获取答案
            answer = chatbot.get_answer(question)
            dic = {
                "answer": answer,
                "origin": 0,
                "is_stay_rule": "false",
                "type": ""
            }
            return_data = json.dumps(dic)

            return Response(return_data, status=200)

    except Exception as e:
        exstr = traceback.format_exc()
        print(exstr)
        return Response(status=500)


# @app.route('/getGraph', methods=['GET', 'POST'])
# def getGraph():
#     try:
#         if request.method == 'POST':
#             post_data = request.get_json()
#             print(post_data)
#             question = post_data.get('question')
#             answer = chatbot.graph_main(question)
#         return json.dumps(answer)
#
#     except Exception as e:
#         return Response(status=500)

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=10000,
        debug=False
    )
    # server = pywsgi.WSGIServer(('0.0.0.0',80),app)
    # server.serve_forever()
    # with open("./questions.txt", "r") as file:
    #     lines = file.readlines()

    # result = chatbot.get_answer("How many off-campus dormitories are there in PolyU")
    # print(result)