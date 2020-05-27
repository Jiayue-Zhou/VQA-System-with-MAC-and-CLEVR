'''
Draft Version
'''

import torch.nn as nn

class Cfgs(object):
    # def __init__(self):
    #     super(Cfgs, self).__init__()
    #    LAYER = 10
        HIDDEN_SIZE = 512
        USE_GLOVE = False
        WORD_EMBED_SIZE = 300
        MEMORY_GATE = False
        SELF_ATTENTION = False
        MAX_STEP = 12
        DROPOUT = 0.15
        FEAT_SIZE = {
            'vqa': {
                'FRCN_FEAT_SIZE': (100, 2048),
                'BBOX_FEAT_SIZE': (100, 5),
            },
            'gqa': {
                'FRCN_FEAT_SIZE': (100, 2048),
                'GRID_FEAT_SIZE': (49, 2048),
                'BBOX_FEAT_SIZE': (100, 5),
            },
            'clevr': {
                'GRID_FEAT_SIZE': (196, 1024),
            },
        }