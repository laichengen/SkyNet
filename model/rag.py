import json
import sys

import numpy as np
import requests
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer, BertModel

# !/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   Embeddings.py
@Time    :   2024/02/10 21:55:39
@Author  :   不要葱姜蒜
@Version :   1.0
@Desc    :   None
'''

import os
from copy import copy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

class BaseEmbeddings:
    """
    Base class for embeddings
    """

    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api

    def get_embedding(self, text: str, model: str) -> List[float]:
        raise NotImplementedError

    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude



class JinaEmbedding(BaseEmbeddings):
    """
    class for Jina embeddings
    """

    def __init__(self, path: str = 'checkpoints/Jina-embeddings-v2-base-zh', is_api: bool = False) -> None:
        super().__init__(path, is_api)
        self._model = self.load_model()

    def get_embedding(self, text: str) -> List[float]:
        text = torch.tensor(text, dtype=torch.long).unsqueeze(0).to(self._model.device)
        embedding = self._model(text)[0][:,0,:].squeeze()
        return embedding.tolist()

    def load_model(self):
        import torch
        from transformers import AutoModel
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(self.path)
        model = AutoModel.from_pretrained(self.path).to(device)
        return model



class Rag():
    def __init__(self,path='checkpoints/Embeddings-bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model = BertModel.from_pretrained(path)
        self.chunk = None

    def get_vector(self,question, chunk):
        self.vectors = []
        encoded_input = self.tokenizer(question, return_tensors='pt')
        output = self.model(**encoded_input)
        cls_embeddings = output.last_hidden_state[:, 0, :].squeeze()
        self.question = cls_embeddings.detach().numpy()
        for doc in tqdm(chunk, desc="Calculating embeddings"):
            encoded_input = self.tokenizer(doc, return_tensors='pt')
            output = self.model(**encoded_input)
            cls_embeddings = output.last_hidden_state[:,0,:].squeeze()
            self.vectors.append(cls_embeddings.detach().numpy())
        return self.vectors

    def get_similarity(self, vector1, vector2):
        return BaseEmbeddings.cosine_similarity(vector1, vector2)

    def query(self, k: int = 1):
        query_vector = self.question
        result = np.array([self.get_similarity(query_vector, vector)
                           for vector in self.vectors])
        result = result.argsort()[-k:].tolist()
        return [self.chunk[i] for i in result]

    def get_chunk(self, text: str, max_token_len: int = 100, cover_content: int = 20):
        chunk_text = []

        curr_len = 0
        curr_chunk = ''

        lines = text.split('\n')  # 假设以换行符分割文本为行

        for line in lines:
            line = line.replace(' ', '')
            line_len = len(self.tokenizer.encode(line))
            if line_len > max_token_len:
                print('warning line_len = ', line_len)
            if curr_len + line_len <= max_token_len:
                curr_chunk += line
                curr_chunk += '\n'
                curr_len += line_len
                curr_len += 1
            else:
                chunk_text.append(curr_chunk)
                curr_chunk = curr_chunk[-cover_content:] + line
                curr_len = line_len + cover_content

        if curr_chunk:
            chunk_text.append(curr_chunk)
        self.chunk = chunk_text
        return chunk_text

    def google_search(self, search_query: str):
        url = "https://google.serper.dev/search"

        payload = json.dumps({"q": search_query})
        headers = {
            'X-API-KEY': '****',
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload).json()
        return response['organic'][0]['snippet']

    def local_search(self, search_query: str):
        f = open('data/rag_knowledge.txt', 'r', encoding='utf-8')
        res = ''
        for line in f.readlines():
            res +=line
        return res

    def answer_with_content(self, llm, content, query):
        prompt = '请根据以下内容：' + content + '回答问题：' + query
        ans = llm.inference(prompt)
        return ans
