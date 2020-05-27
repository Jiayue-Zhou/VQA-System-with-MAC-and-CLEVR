'''
Draft Version
'''

import flask
import numpy as np
from flask import Flask, jsonify, request
import torch
import torch.nn as nn
import os, json
import time
from clevr_data_provider_layer import proc_ques, extract_feat, tokenize, ans_stat
from model.net import Net

VDICT_PATH = ""
ADICT_PATH = ""


MY_MODEL_PKL = ""

#global variables
net_vqa = None
ques_ix = None
max_token = None
token_to_ix = None
ix_to_ans = None

app = Flask(__name__, static_url_path='')


feature_path = ["feats/000.npz", "feats/051.npz", "feats/116.npz",
                "feats/162.npz", "feats/212.npz", "feats/258.npz",
                "feats/308.npz", "feats/357.npz", "feats/403.npz",
                "feats/452.npz"]


def load_model_vqa(map_location):
    global ques_ix
    global net_vqa
    global max_token
    global token_to_ix
    global ix_to_ans
    model_path = 'CKPT/epoch19.pkl'
    train_path = 'data/CLEVR_train_questions.json'
    print("Load model...")
    state_dict = torch.load(model_path, map_location = map_location)['state_dict']
    ques_stat = json.load(open(train_path, 'r'))['questions']
    stat_ans = json.load(open(train_path, 'r'))['questions']

    token_to_ix, pretrained_emb, max_token = tokenize(ques_stat, False)
    ans_to_ix, ix_to_ans = ans_stat(stat_ans)



    ans_size = ans_to_ix.__len__()
    token_size = token_to_ix.__len__()

    print("token_size:", token_size)
    print("ans_size:", ans_size)
    net_vqa = Net(pretrained_emb, token_size, ans_size)
    net_vqa.load_state_dict(state_dict)
    net_vqa.eval()




def setup():
    global net_vqa
    global token_to_ix
    global ix_to_ans


# routes
@app.route('/', methods = ['GET'])
def index():
    return app.send_static_file('demo2.html')


@app.route("/predict", methods=['POST'])
def predict():
    global ques_ix
    global max_token
    global token_to_ix
    #result = {"success": False}
    img_hash = request.form['img_id']
    #if img_hash not in feature_cache:
    #    return jsonify({'error': 'Unknown image ID. Try uploading the image again.'})
    time_start = time.time()
    print("Predict...")
    img_hash = int(img_hash)
    img_feature = feature_path[img_hash]
    img_feature = np.load(img_feature)['x']
    img_feature = torch.from_numpy(img_feature)
    question = request.form['question']
    print("Get question:", question)

    ques_ix = proc_ques(question, token_to_ix, max_token)
    ques_ix = torch.from_numpy(ques_ix)
    print("ques_ix:", ques_ix.shape)
    imgfeat_batch = img_feature.unsqueeze(0)
    quesix_batch = ques_ix.unsqueeze(0)
    print("imgshape:", imgfeat_batch.shape)
    print("quesshape:", ques_ix.shape)
    pred = net_vqa(imgfeat_batch, quesix_batch)
    print("pred:", pred)
    time_end = time.time()
    pred_np = pred.cpu().data.numpy()
    print("pred_np:", pred_np)
    pred_argmax = np.argmax(pred_np, axis=1)[0]
    print("pred_argmax:", pred_argmax)
    pred_ans = ix_to_ans[pred_argmax]
    print("predict done. pred_ans:", pred_ans)
    timegap = time_end - time_start


    #vqa_net.forward()
    #scores = vqa_net.blobs['prediction'].data.flatten()

    # attention visualization
    #att_map = vqa_net.blobs['att_map0'].data.copy()[0]
    #source_img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_hash + '.jpg')
    #path0 = save_attention_visualization(source_img_path, att_map, img_ques_hash)

    #scores = softmax(scores)
    #op_indices = scores.argsort()[::-1][:5]
    #top_answers = [vqa_data_provider.vec_to_answer(i) for i in top_indices]
    #top_scores = [float(scores[i]) for i in top_indices]

    json = {'answer': pred_ans,
        #'answers': top_answers,
        #'scores': top_scores,
        #'viz': [path0],
        'time': timegap}
    return jsonify(json)




if __name__ == '__main__':
    setup()
    map_location = 'cpu'
    load_model_vqa(map_location)
    app.run(host = '127.0.0.1', port = 5000)

