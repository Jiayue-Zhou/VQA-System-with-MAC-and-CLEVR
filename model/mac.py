'''
Draft Version
'''

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
#from model.configs import Cfgs

def linear(in_dim, out_dim, bias = True):
    lin = nn.Linear(in_dim, out_dim, bias = bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()
    return lin



class ControlUnit(nn.Module):
    def __init__(self, dim, max_step):
        super().__init__()

        self.position_aware = nn.ModuleList()
        for i in range(max_step):
            self.position_aware.append(linear(dim * 2, dim))

        self.control_question = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

        self.dim = dim
    def forward(self, step, context, question, control):
        position_aware = self.position_aware[step](question)

        control_question = torch.cat([control, position_aware], 1)
        control_question = self.control_question(control_question)
        control_question = control_question.unsqueeze(1)

        context_prod = control_question * context
        attn_weight = self.attn(context_prod)

        attn = F.softmax(attn_weight, 1)

        next_control = (attn * context).sum(1)

        return next_control

class ReadUnit(nn.Module):
        def __init__(self, dim):
            super().__init__()

            self.mem = linear(dim, dim)
            self.concat = linear(dim * 2, dim)
            #self.know = linear(dim, dim)
            self.attn = linear(dim, 1)

        def forward(self, memory, know, control):
            mem = self.mem(memory[-1]).unsqueeze(2)
            #know = self.know(know)
            concat = self.concat(torch.cat([mem * know, know], 1) \
                                 .permute(0, 2, 1))

            attn = concat * control[-1].unsqueeze(1)
            attn = self.attn(attn).squeeze(2)
            attn = F.softmax(attn, 1).unsqueeze(1)

            read = (attn * know).sum(2)

            return read

class WriteUnit(nn.Module):
    def __init__(self, dim, self_attention = False, memory_gate = False):
        super().__init__()

        self.concat = linear(dim * 2, dim)

        if self_attention:
            self.attn = linear(dim, 1)
            self.mem = linear(dim, dim)

        if memory_gate:
            self.control = linear(dim, 1)

        self.self_attention = self_attention
        self.memory_gate = memory_gate

    def forward(self, memories, retrieved, controls):
        prev_mem = memories[-1]
        concat = self.concat(torch.cat([retrieved, prev_mem], 1))
        next_mem = concat

        if self.self_attention:
            controls_cat = torch.stack(controls[:-1], 2)
            attn = controls[-1].unsqueeze(2) * controls_cat
            attn = self.attn(attn.permute(0, 2, 1))
            attn = F.softmax(attn, 1).permute(0, 2, 1)

            memories_cat = torch.stack(memories, 2)
            attn_mem = (attn * memories_cat).sum(2)
            next_mem = self.mem(attn_mem) + concat

        if self.memory_gate:
            control = self.control(controls[-1])
            gate = F.sigmoid(control)
            next_mem = gate * prev_mem + (1 - gate) * next_mem

        return next_mem

class MACUnit(nn.Module):
    def __init__(self, Cfgs):
        super().__init__()

        self.control = ControlUnit(Cfgs.HIDDEN_SIZE, Cfgs.MAX_STEP)
        self.read = ReadUnit(Cfgs.HIDDEN_SIZE)
        self.write = WriteUnit(Cfgs.HIDDEN_SIZE, Cfgs.SELF_ATTENTION, Cfgs.MEMORY_GATE)

        self.mem_0 = nn.Parameter(torch.zeros(1, Cfgs.HIDDEN_SIZE))
        self.control_0 = nn.Parameter(torch.zeros(1, Cfgs.HIDDEN_SIZE))

        self.dim = Cfgs.HIDDEN_SIZE
        self.max_step = Cfgs.MAX_STEP
        self.dropout = Cfgs.DROPOUT

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)

        return mask

    def forward(self, context, question, knowledge):
        b_size = question.size(0)

        control = self.control_0.expand(b_size, self.dim)
        memory = self.mem_0.expand(b_size, self.dim)

        if self.training:
            control_mask = self.get_mask(control, self.dropout)
            memory_mask = self.get_mask(memory, self.dropout)
            control = control * control_mask
            memory = memory * memory_mask

        controls = [control]
        memories = [memory]

        for i in range(self.max_step):
            control = self.control(i, context, question, control)
            if self.training:
                control = control * control_mask
            controls.append(control)

            read = self.read(memories, knowledge, controls)
            memory = self.write(memories, read, controls)
            if self.training:
                memory = memory * memory_mask
            memories.append(memory)

        return memory

