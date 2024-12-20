"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Author: xinyi
@file:bge_m3.py
@date: 2024/12/19 14:45
@description: bge-m3使用
@refer:https://github.com/FlagOpen/FlagEmbedding
"""
import sys
import torch, os
from transformers import AutoTokenizer, AutoModel
from FlagEmbedding import BGEM3FlagModel

model_path = "E:/PrivateWork/LLM/LLM_Backend/BAAI/bge-m3"

#  model structure XLMRobertaModel
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# raw_model = AutoModel.from_pretrained(model_path)
# print(raw_model.eval())

model = BGEM3FlagModel(model_path, use_fp16=True)

sentences_1 = ["BGE M3的特性是什么?", "BM25编码有什么特点"]
sentences_2 = [
    "BGE-M3 是一个混合检索模型，支持稠密检索（Dense Retrieval），还支持稀疏检索（Sparse Retrieval）与多向量检索",
    "BM25 是一个词袋检索模型， 通过与query词重复比例进行检索排序"]

output_1 = model.encode(sentences_1, max_length=100, return_dense=True, return_sparse=True, return_colbert_vecs=True)
output_2 = model.encode(sentences_2, max_length=100, return_dense=True, return_sparse=True, return_colbert_vecs=True)

################ Test1: Dense Retrival ################
dense_emb_1 = output_1['dense_vecs']  # 2 * 1024
dense_emb_2 = output_2['dense_vecs']
s_dense = dense_emb_1 @ dense_emb_2.T
print(f"---s_dense: {s_dense}")

################ Test2: lexical Retrival ################
# you can see the weight for each token:
print(model.convert_id_to_token(output_1['lexical_weights']))

s_lex_10_20 = model.compute_lexical_matching_score(output_1['lexical_weights'][0], output_2['lexical_weights'][0])
s_lex_10_21 = model.compute_lexical_matching_score(output_1['lexical_weights'][0], output_2['lexical_weights'][1])
s_lex_11_20 = model.compute_lexical_matching_score(output_1['lexical_weights'][1], output_2['lexical_weights'][0])
s_lex_11_21 = model.compute_lexical_matching_score(output_1['lexical_weights'][1], output_2['lexical_weights'][1])
print(f"---s_lex: {s_lex_10_20}, {s_lex_10_21}, {s_lex_11_20}, {s_lex_11_21}")

################ Test3: Multi-Vector Retrival ################
s_mul_10_20 = model.colbert_score(output_1['colbert_vecs'][0], output_2['colbert_vecs'][0]).item()
s_mul_10_21 = model.colbert_score(output_1['colbert_vecs'][0], output_2['colbert_vecs'][1]).item()
s_mul_11_20 = model.colbert_score(output_1['colbert_vecs'][1], output_2['colbert_vecs'][0]).item()
s_mul_11_21 = model.colbert_score(output_1['colbert_vecs'][1], output_2['colbert_vecs'][1]).item()
print(f"---s_mul: {s_mul_10_20}, {s_mul_10_21}, {s_mul_11_20}, {s_mul_11_21}")

# ################ Test4: 混合检索 Retrival ################
s_rank_10_20 = 1 / 3 * s_dense[0][0] + 1 / 3 * s_lex_10_20 + 1 / 3 * s_mul_10_20
s_rank_10_21 = 1 / 3 * s_dense[0][1] + 1 / 3 * s_lex_10_21 + 1 / 3 * s_mul_10_21
s_rank_11_20 = 1 / 3 * s_dense[1][0] + 1 / 3 * s_lex_11_20 + 1 / 3 * s_mul_11_20
s_rank_11_21 = 1 / 3 * s_dense[1][1] + 1 / 3 * s_lex_11_21 + 1 / 3 * s_mul_11_21

print("row1, first sentence 与其他的句子相似分数", s_rank_10_20, s_rank_10_21)
print("row2, second sentence 与其他的句子相似分数", s_rank_11_20, s_rank_11_21)
