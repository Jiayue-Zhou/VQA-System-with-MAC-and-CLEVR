'''
Draft Version
'''

import json
import torch
import torch.nn as nn
import re
import requests
import numpy as np


REST_API_URL = 'http://127.0.0.1:5000/predict'

def predict_result(image_path, question, language):
    payload = {'image_path': image_path, 'question': question, 'language': language}

    r = requests.post(REST_API_URL, data = payload).json()

    if r['success']:
        result = r['predictions']
        pred_ans = result['pred_ans']
        chinese_ans = result['chinese_ans']
        language = result['language']
        print('pred_ans:', pred_ans)
        if language in ['ZH', 'zh']:
            print('chinese_ans:', chinese_ans)

    else:
        print('Request failed')

def extract_feat(image_path):
    number = image_path[-3:]
    feats_path = "/feats/" + number + ".npz"
    feat_np = np.load(feats_path)
    feats = torch.from_numpy(feat_np['x'])
    return feats


def tokenize(stat_ques_list, use_glove):
    token_to_ix = {
        'PAD': 0,
        'UNK': 1,
        'CLS': 2,
    }

    spacy_tool = None
    pretrained_emb = []
    #if use_glove:
    #    spacy_tool = en_vectors_web_lg.load()
    #    pretrained_emb.append(spacy_tool('PAD').vector)
    #    pretrained_emb.append(spacy_tool('UNK').vector)
    #    pretrained_emb.append(spacy_tool('CLS').vector)

    max_token = 0
    for ques in stat_ques_list:
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['question'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        if len(words) > max_token:
            max_token = len(words)

        for word in words:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)
                if use_glove:
                    pretrained_emb.append(spacy_tool(word).vector)

    pretrained_emb = np.array(pretrained_emb)

    return token_to_ix, pretrained_emb, max_token


def ans_stat(stat_ans_list):
    ans_to_ix = {}
    ix_to_ans = {}

    for ans_stat in stat_ans_list:
        ans = ans_stat['answer']

        if ans not in ans_to_ix:
            ix_to_ans[ans_to_ix.__len__()] = ans
            ans_to_ix[ans] = ans_to_ix.__len__()

    return ans_to_ix, ix_to_ans

def proc_ques(ques, token_to_ix, max_token):
    ques_ix = np.zeros(max_token, np.int64)
    words = re.sub(r"[.,'!?\"()*#:;]",'',ques.lower()
                   ).replace('-', ' ').replace('/', ' ').split()
    for ix, word in enumerate(words):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix == max_token - 1:
            break

    return ques_ix



