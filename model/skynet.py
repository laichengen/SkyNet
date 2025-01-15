import ast
import contextlib
import json
import sys
from io import StringIO

import requests
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from model.basemodel import BaseModel
from model.rag import Rag
from model.rlhf.train_ppo import train_ppo
from model.rlhf.train_rm import train_rm
from model.utils import extract_code, stdoutIO, run_code, is_right


# def learn_by_human():
#     print('通过人类来辅助学习')


class SkyNet:
    def __init__(self):
        self.name = '天网'
        self.basemodel = BaseModel()
        self.rag = Rag()
        self.learn_from_web = False

    def learn_by_rag(self, question, learn_from_local=True):
        if learn_from_local:
            crawl_data = self.rag.local_search(question)
        else:
            crawl_data = self.rag.google_search(question)
        chunk_data = self.rag.get_chunk(crawl_data)
        self.rag.get_vector(question, chunk_data)
        content = self.rag.query()
        content = '\n'.join(content)
        # content = '冲量的计算公式是力的作用时间*力的大小,有较强的上进心和学习能力，为激励其今后更加奋发，做出更出色的成绩和成果'
        # ans = self.rag.answer_with_content(self.basemodel, content, question)
        return content

    def is_caculate(self, question):
        prompt = "请你识别下面问题是否可以计算" + question
        res = self.basemodel.inference(prompt)
        if res.find('不') != -1:
            return False
        else:
            return True

    # def illustrate(self, text, answer):
    #     prompt_text = "请根据以下信息："
    #     prompt_text += self.learn_by_rag(text, True)
    #     prompt_text = prompt_text + text
    #     generate_code = []
    #     while True:
    #         prompt_text = "写一个算法：" + prompt_text + "。写一个算法，结果以列表的形式返回："
    #         res = self.basemodel.inference(prompt_text)  # 生成代码
    #         try:
    #             res, code = run_code(res)  # 运行代码
    #             print(code)
    #             print("out:", res)
    #         except:
    #             continue
    #         if answer is None or is_right(generate_code, res, answer, code):
    #             break
    #     if len(generate_code) > 1:
    #         samples = [{
    #             "question": text,
    #             "positive": generate_code[-1],
    #             "negative": generate_code[-2]
    #         }]
    #         open('caches/data/rm_train.json', 'w', encoding='utf-8').write(
    #             json.dumps(samples, ensure_ascii=False))

    def inference(self, text=None, is_correct=False):
        # 判断是否开启代码模式
        is_code = self.is_caculate(text)
        print("代码模式:" + str(is_code))
        # 使用rag技术进行自我学习
        prompt_text = ''
        prompt_text += "请根据以下信息："
        prompt_text += self.learn_by_rag(text)
        prompt_text = prompt_text + text
        generate_code = []
        if is_code:
            prompt_text = "写一个算法：" + prompt_text + "。写一个算法，结果以列表的形式返回："
            # 通过将任务的自然语言描述转成编程语言，来逐步进行任务的求解
            while True:
                res = self.basemodel.inference(prompt_text)  # 生成代码
                generate_code.append(res)
                try:
                    res, code = run_code(res)  # 运行代码
                    print(code)
                    print("out:", res)
                    break
                except:
                    continue
            if len(generate_code) > 1:
                samples = [{
                    "question": text,
                    "positive": generate_code[-1],
                    "negative": generate_code[-2]
                }]
                open('data/rm_train.json', 'w', encoding='utf-8').write(
                    json.dumps(samples, ensure_ascii=False))
                # 使用RLHF来进行自我纠正
                if is_correct:
                    model_path = "checkpoints/GPT2"
                    train_path = 'data/rm_train.json'
                    train_rm(train_path, model_path)
                    train_ppo(train_path, model_path)

        else:
            res = self.basemodel.inference(prompt_text)
            print(res)
        return res
