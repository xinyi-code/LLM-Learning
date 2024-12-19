"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Author: xinyi
@file:bge_m3.py
@date: 2024/12/19 14:45
@description: bge-m3使用
@refer:https://github.com/FlagOpen/FlagEmbedding
"""

from transformers import AutoTokenizer, AutoModel
import torch, os

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
raw_model = AutoModel.from_pretrained("BAAI/bge-m3")

#  model structure XLMRobertaModel
print(raw_model.eval())

from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

sentences_1 = ["BGE M3的特性是什么?", "BM25编码有什么特点"]
sentences_2 = [
    "BGE-M3 是一个混合检索模型，支持稠密检索（Dense Retrieval），还支持稀疏检索（Sparse Retrieval）与多向量检索",
    "BM25 是一个词袋检索模型， 通过与query词重复比例进行检索排序"]

################ Test1: Dense Retrival ################
embeddings_1 = model.encode(sentences_1, max_length=10)['dense_vecs']
embeddings_2 = model.encode(sentences_2, max_length=100)['dense_vecs']

# compute the similarity scores
s_dense = embeddings_1 @ embeddings_2.T
print(s_dense)

output_1 = model.encode(sentences_1, return_sparse=True)
output_2 = model.encode(sentences_2, return_sparse=True)

# you can see the weight for each token:
print(model.convert_id_to_token(output_1['lexical_weights']))

# compute the scores via lexical mathcing
s_lex_10_20 = model.compute_lexical_matching_score(output_1['lexical_weights'][0], output_2['lexical_weights'][0])
s_lex_10_21 = model.compute_lexical_matching_score(output_1['lexical_weights'][0], output_2['lexical_weights'][1])

print(s_lex_10_20)
print(s_lex_10_21)
