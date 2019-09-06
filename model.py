import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import pdb
import math
import itertools
from copy import deepcopy
from masked_cel import *
from anytree import Node


class AttnDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, dictionary, dropout):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        #self.input_size = input_size
        self.char_vocab_size = len(dictionary['char']) + 1
        self.pos1_vocab_size = len(dictionary['pos1']) + 1
        self.pos2_vocab_size = len(dictionary['pos2']) + 1
        self.form_vocab_size = len(dictionary['form']) + 1
        self.input_size = self.char_vocab_size
        #self.char_embed = nn.Embedding(self.char_vocab_size, input_size)
        self.char_embed = nn.Embedding(self.char_vocab_size, self.char_vocab_size)
        self.char_embed.from_pretrained(torch.eye(self.char_vocab_size))
        self.pos1_embed = nn.Embedding(self.pos1_vocab_size, hidden_size)
        self.pos2_embed = nn.Embedding(self.pos2_vocab_size, hidden_size)
        self.form_embed = nn.Embedding(self.form_vocab_size, hidden_size)
        self.encoder = nn.LSTM(self.input_size, hidden_size)
        self.decoder = nn.LSTM(self.input_size + hidden_size, hidden_size)
        self.key_size = 50
        self.q_key = nn.Linear(hidden_size, self.key_size)
        self.q_key_2 = nn.Linear(hidden_size, self.key_size)
        self.q_value = nn.Linear(input_size, hidden_size)
        self.a_key = nn.Linear(hidden_size, self.key_size)
        self.p_key = nn.Linear(hidden_size, self.key_size)
        self.hidden_fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.hidden_fc2 = nn.Linear(hidden_size, hidden_size)
        self.cell_fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.cell_fc2 = nn.Linear(hidden_size, hidden_size)
        self.word_dist = nn.Linear(self.input_size, self.char_vocab_size)
        self.word_dist.weight = self.char_embed.weight
        self.out = nn.Linear(hidden_size * 3 + self.input_size * 0, self.char_vocab_size)
        self.bert_map = nn.Linear(768, self.hidden_size)
        self.eou = 2
        self.drop = nn.Dropout(dropout)

    def post_parameters(self):
        return []

    def flatten_parameters(self):
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()

    def hidden_transform(self, hidden):
        return self.hidden_fc2(F.relu(self.hidden_fc1(hidden)))

    def cell_transform(self, cell):
        return self.cell_fc2(F.relu(self.cell_fc1(cell)))

    def init_hidden(self, src_hidden):
        hidden = src_hidden
        cell = src_hidden
        hidden = self.hidden_transform(hidden)
        cell = self.cell_transform(cell)
        return (hidden, cell)

    def attention(self, decoder_hidden, src_hidden, src_lengths, bern=False):
        a_key = self.a_key(decoder_hidden[0].squeeze(0))
        length = src_hidden.size(1)

        if bern:
            q_key = self.q_key_2(src_hidden)
        else:
            q_key = self.q_key(src_hidden)
        q_value = src_hidden

        q_energy = torch.bmm(q_key, a_key.unsqueeze(2)).squeeze(2)
        q_mask = torch.arange(length).long().cuda().repeat(src_hidden.size(0), 1) < torch.cuda.LongTensor(src_lengths).repeat(length, 1).transpose(0, 1)
        q_energy[~q_mask] = -np.inf
        #q_weights = F.sigmoid(q_energy).unsqueeze(1)
        q_weights = F.softmax(q_energy, dim=1).unsqueeze(1)
        q_context = torch.bmm(q_weights, q_value)

        return q_context


    def forward(self, src_seqs, src_lengths, trg_seqs, trg_lengths, pos_seqs, form_seqs, bert_seqs):
        batch_size = src_seqs.size(0)
        form_seqs, form_lengths = form_seqs
        bert_seqs = torch.from_numpy(np.vstack(bert_seqs)).cuda()
        bert_seqs = self.bert_map(bert_seqs.float())

        src_embed = self.drop(self.char_embed(src_seqs.cuda()))
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)

        length = src_hidden.size(1)
        ans_embed = self.drop(self.char_embed(trg_seqs.cuda())).transpose(0, 1)
        trg_l = ans_embed.size(0)

        decoder_input = ans_embed[[0], :, :]
        decoder_outputs = torch.FloatTensor(batch_size, trg_l - 1, self.char_vocab_size).cuda()
        
        decoder_hidden = self.init_hidden(torch.cat((src_last_hidden[0], bert_seqs.unsqueeze(0)), dim=2))
        pos_seqs = torch.cuda.LongTensor(pos_seqs)
        form_embed = torch.cat((self.pos1_embed(pos_seqs[:, [0]]),
                                self.pos2_embed(pos_seqs[:, [1]]),
                                self.form_embed(form_seqs.cuda())),
                               dim=1)
        form_embed = self.drop(form_embed)
        form_lengths = [l + 2 for l in form_lengths]
        form_input = form_embed.sum(1).unsqueeze(1)
        decoder_input = torch.cat((decoder_input, self.drop(form_input).transpose(0, 1)), dim=2)
        for step in range(trg_l - 1):
            context = self.attention(decoder_hidden, src_hidden, src_lengths)
            form_context = self.attention(decoder_hidden, form_embed, form_lengths, bern=True)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_output = torch.cat((self.drop(decoder_output.transpose(0, 1)), self.drop(context), self.drop(form_context)), dim=2)
            #decoder_output = torch.cat((self.drop(decoder_output.transpose(0, 1)), self.drop(context)), dim=2)
            #decoder_outputs[:, step, :] = self.word_dist(F.tanh(self.out(decoder_output.squeeze(1))))
            decoder_outputs[:, step, :] = self.out(decoder_output.squeeze(1))
            #decoder_input = ans_embed[[step+1], :, :]
            decoder_input = torch.cat((ans_embed[[step+1], :, :], self.drop(form_input).transpose(0, 1)), dim=2)

        return decoder_outputs

    def generate(self, src_seqs, src_lengths, pos_seqs, form_seqs, bert_seqs, max_len, beam_size, top_k):
        form_seqs, form_lengths = form_seqs
        bert_seqs = torch.from_numpy(np.vstack(bert_seqs)).cuda()
        bert_seqs = self.bert_map(bert_seqs.float())
        src_embed = self.char_embed(src_seqs.cuda())
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(torch.cat((src_last_hidden[0], bert_seqs.unsqueeze(0)), dim=2))
        pos_seqs = torch.cuda.LongTensor(pos_seqs)
        form_embed = torch.cat((self.pos1_embed(pos_seqs[:, [0]]),
                                self.pos2_embed(pos_seqs[:, [1]]),
                                self.form_embed(form_seqs.cuda())),
                               dim=1)
        _batch_size = src_embed.size(0)
        assert _batch_size == 1
        eos_filler = torch.zeros(beam_size).long().cuda().fill_(self.eou)
        decoder_input = self.char_embed(torch.zeros(_batch_size).long().cuda().fill_(1)).unsqueeze(1)
        length = src_hidden.size(1)

        form_embed = self.drop(form_embed)
        form_lengths = [l + 2 for l in form_lengths]
        form_input = form_embed.sum(1).unsqueeze(1)
        decoder_input = torch.cat((decoder_input, self.drop(form_input).transpose(0, 1)), dim=2)

        context = self.attention(decoder_hidden, src_hidden, src_lengths)
        form_context = self.attention(decoder_hidden, form_embed, form_lengths, bern=True)
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        #decoder_output = torch.cat((decoder_output, context, form_context), dim=2)
        decoder_output = torch.cat((self.drop(decoder_output.transpose(0, 1)), self.drop(context), self.drop(form_context)), dim=2)
        #decoder_output = self.word_dist(F.tanh(self.out(decoder_output.squeeze(1))))
        decoder_output = self.out(decoder_output.squeeze(1))
        decoder_output[:, 0] = -np.inf

        logprobs, argtop = torch.topk(F.log_softmax(decoder_output, dim=1), beam_size, dim=1)
        beam = Variable(torch.zeros(_batch_size, beam_size, max_len)).long().cuda()
        beam[:, :, 0] = argtop
        beam_probs = logprobs.clone()
        beam_eos = (argtop == self.eou).data
        decoder_hidden = (decoder_hidden[0].expand(1, beam_size, self.hidden_size).contiguous(),
                          decoder_hidden[1].expand(1, beam_size, self.hidden_size).contiguous())
        src_last_hidden = (src_last_hidden[0].expand(1, beam_size, self.hidden_size).contiguous(),
                           src_last_hidden[1].expand(1, beam_size, self.hidden_size).contiguous())
        form_input = form_input.expand(beam_size, 1, self.hidden_size).contiguous()
        form_embed = form_embed.expand(beam_size, form_embed.size(1), self.hidden_size).contiguous()
        decoder_input = self.char_embed(argtop.view(-1)).unsqueeze(1)
        decoder_input = torch.cat((decoder_input, form_input), dim=2).transpose(0, 1)
        src_hidden = src_hidden.expand(beam_size, length, self.hidden_size)

        for t in range(max_len-1):
            
            context = self.attention(decoder_hidden, src_hidden, src_lengths)
            form_context = self.attention(decoder_hidden, form_embed, form_lengths, bern=True)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_output = torch.cat((decoder_output.transpose(0, 1), context, form_context), dim=2)
            decoder_output = self.out(decoder_output.squeeze(1))

            logprobs, argtop = torch.topk(F.log_softmax(decoder_output, dim=1), top_k, dim=1)
            #best_probs, best_args = (beam_probs.repeat(top_k, 1).transpose(0, 1) + logprobs).view(-1).topk(beam_size)
            #best_probs, best_args = (beam_probs.view(-1).unsqueeze(1).expand(_batch_size * beam_size, top_k) + logprobs).view(_batch_size, beam_size, -1).view(_batch_size, -1).topk(beam_size)
            total_probs = beam_probs.view(-1).unsqueeze(1).expand(_batch_size * beam_size, top_k).contiguous()
            total_probs[~beam_eos[0]] = (total_probs[~beam_eos[0]] * (t + 1) + logprobs[~beam_eos[0]]) / (t + 2)
            total_probs[beam_eos[0], 1:] = -np.inf
            best_probs, best_args = total_probs.contiguous().view(_batch_size, -1).topk(beam_size, dim=1)

            decoder_hidden = (decoder_hidden[0].view(1, _batch_size, beam_size, -1),
                              decoder_hidden[1].view(1, _batch_size, beam_size, -1))
            for x in range(_batch_size):
                last = (best_args / top_k)[x]
                curr = (best_args % top_k)[x]
                beam[x, :, :] = beam[x][last, :]
                beam_eos[x, :] = beam_eos[x][last.data]
                beam_probs[x, :] = beam_probs[x][last.data]
                beam[x, :, t+1] = argtop.view(_batch_size, beam_size, top_k)[x][last.data, curr.data] * Variable(~beam_eos[x]).long() + eos_filler * Variable(beam_eos[x]).long()
                mask = torch.cuda.ByteTensor(_batch_size, beam_size).fill_(0)
                mask[x] = ~beam_eos[x]
                #beam_probs[mask] = (beam_probs[mask] * (t+1) + best_probs[mask]) / (t+2)
                beam_probs[mask] = best_probs[mask]
                decoder_hidden[0][:, x, :, :] = decoder_hidden[0][:, x, :, :][:, last, :]
                decoder_hidden[1][:, x, :, :] = decoder_hidden[1][:, x, :, :][:, last, :]
            beam_eos = beam_eos | (beam[:, :, t+1] == self.eou).data
            decoder_hidden = (decoder_hidden[0].view(1, _batch_size * beam_size, -1),
                              decoder_hidden[1].view(1, _batch_size * beam_size, -1))
            decoder_input = self.char_embed(beam[:, :, t+1].contiguous().view(-1)).unsqueeze(1)
            decoder_input = torch.cat((decoder_input, form_input), dim=2).transpose(0, 1)
            if beam_eos.all():
                break
        
        best, best_arg = beam_probs.max(1)
        generations = beam[torch.arange(_batch_size).long().cuda(), best_arg.data].data.cpu()
        return generations, best

    def masked_loss(self, logits, target, lengths, mask):
        batch_size = logits.size(0)
        max_len = lengths.data.max()
        #max_len = logits.size(1)
        l_mask  = torch.arange(max_len).long().cuda().expand(batch_size, max_len) < lengths.data.expand(max_len, batch_size).transpose(0, 1)
        log_probs_flat = F.log_softmax(logits, dim=2).view(-1, logits.size(-1))
        target_flat = target.contiguous().view(-1, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        losses = losses_flat.view(*target.size())
        _mask = Variable(l_mask * mask)
        losses = losses * _mask.float()
        loss = losses.sum() / _mask.float().sum()
        return loss, _mask.float().sum()

    def loss(self, src_seqs, src_lengths, trg_seqs, trg_lengths, pos_seqs, form_seqs, bert_seqs):
        decoder_outputs = self.forward(src_seqs, src_lengths, trg_seqs, trg_lengths, pos_seqs, form_seqs, bert_seqs)
        batch_size = decoder_outputs.size(0)
        loss = compute_perplexity(decoder_outputs.squeeze(1), trg_seqs[:, 1:].cuda(), torch.cuda.LongTensor(trg_lengths) - 1)
        return loss
        

def split_heads(x, num_heads):
    """split x into multi heads
    Args:
        x: [batch_size, length, depth]
    Returns:
        y: [[batch_size, length, depth / num_heads] x heads]
    """
    sz = x.size()
    # x -> [batch_size, length, heads, depth / num_heads]
    x = x.view(sz[0], sz[1], num_heads, sz[2] // num_heads)
    # [batch_size, length, 1, depth // num_heads] * 
    heads = torch.chunk(x, num_heads, 2)
    x = []
    for i in range(num_heads):
        x.append(torch.squeeze(heads[i], 2))
    return x


def combine_heads(x):
    """combine multi heads
    Args:
        x: [batch_size, length, depth / num_heads] x heads
    Returns:
        x: [batch_size, length, depth]
    """
    return torch.cat(x, 2)
    

def dot_product_attention(q, k, v, bias, edge_bias, edge_val, dropout, to_weights=False):
    """dot product for query-key-value
    Args:
        q: query antecedent, [batch, length, depth]
        k: key antecedent,   [batch, length, depth]
        v: value antecedent, [batch, length, depth]
        bias: masked matrix
        dropout: dropout rate
        to_weights: whether to print weights
    """
    # [batch, length, depth] x [batch, depth, length] -> [batch, length, length]
    logits = torch.bmm(q, k.transpose(1, 2).contiguous())
    if bias is not None:
        logits += bias
    if edge_bias is not None:
        src_len = logits.size(1)
        key_size = q.size(-1)
        #rel_pos_logits = torch.bmm(q.unsqueeze(2).view(-1, 1, key_size), edge_bias.transpose(2, 3).contiguous().view(-1, key_size, src_len))
        rel_pos_logits = (q[:, :, None, :] * edge_bias).sum(3)
        rel_pos_logits = rel_pos_logits.view(-1, src_len, 1, src_len).squeeze(2)
        logits = logits + rel_pos_logits
    size = logits.size()
    weights = F.softmax(logits.view(size[0] * size[1], size[2]), dim=1)
    weights = weights.view(size)
    value = torch.bmm(weights, v)
    if edge_val is not None:
        src_len = weights.size(1)
        #rel_pos_value = torch.bmm(weights.unsqueeze(1).view(-1, 1, src_len), edge_val.view(-1, src_len, edge_val.size(-1)))
        rel_pos_value = (weights.unsqueeze(3) * edge_val).sum(2)
        rel_pos_value = rel_pos_value.view(-1, src_len, edge_val.size(-1))
        value = value + rel_pos_value
    if to_weights:
        return value, weights
    else:
        return value


class MultiheadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, 
                 key_depth, value_depth, output_depth,
                 num_heads, dropout=0.1):
        super(MultiheadAttention, self).__init__()

        self._query = nn.Linear(key_depth, key_depth, bias=False)
        self._key = nn.Linear(key_depth, key_depth, bias=False)
        self._value = nn.Linear(value_depth, value_depth, bias=False)
        self.output_perform = nn.Linear(value_depth, output_depth, bias=False)

        self.num_heads = num_heads
        self.key_depth_per_head = key_depth // num_heads
        self.dropout = dropout
        
    def forward(self, query_antecedent, memory_antecedent, bias, edge_bias, edge_val, to_weights=False):
        if memory_antecedent is None:
            memory_antecedent = query_antecedent
        q = self._query(query_antecedent)
        k = self._key(memory_antecedent)
        v = self._value(memory_antecedent)
        q *= self.key_depth_per_head ** -0.5
        
        # split heads
        q = split_heads(q, self.num_heads)
        k = split_heads(k, self.num_heads)
        v = split_heads(v, self.num_heads)

        x = []
        avg_attn_scores = None
        for i in range(self.num_heads):
            results = dot_product_attention(q[i], k[i], v[i],
                                            bias,
                                            edge_bias,
                                            edge_val,
                                            self.dropout,
                                            to_weights)
            if to_weights:
                y, attn_scores = results
                if avg_attn_scores is None:
                    avg_attn_scores = attn_scores
                else:
                    avg_attn_scores.add_(attn_scores)
            else:
                y = results
            x.append(y)
        x = combine_heads(x)
        x = self.output_perform(x)
        if to_weights:
            return x, avg_attn_scores / self.num_heads
        else:
            return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(hidden_size, filter_size, bias=False)
        self.fc2 = nn.Linear(filter_size, hidden_size, bias=False)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


def residual(x, y, dropout, training):
    """Residual connection"""
    y = F.dropout(y, p=dropout, training=training)
    return x + y


class Transformer(nn.Module):
    def __init__(self, input_size, rnn_hidden_size, att_hidden_size, word_vectors, dictionary, dropout, num_layers, num_heads):
        super(Transformer, self).__init__()
        self.rnn_hidden_size = rnn_hidden_size
        self.att_outer_size = 512
        self.att_hidden_size = att_hidden_size
        self.input_size = input_size
        self.word_vocab_size = len(dictionary['word']) + 1
        self.pos_vocab_size = len(dictionary['pos']) + 1
        self.dep_vocab_size = len(dictionary['dep']) + 1
        self.word_embed = nn.Embedding(self.word_vocab_size, input_size)
        self.word_embed_map = nn.Linear(input_size, self.att_outer_size)
        self.pos_embed = nn.Embedding(self.pos_vocab_size, self.att_outer_size)
        self.dep_embed = nn.Embedding(self.dep_vocab_size, self.att_outer_size)
        self.dep_key_embed = nn.Embedding(self.dep_vocab_size, self.att_outer_size // num_heads)
        self.dep_val_embed = nn.Embedding(self.dep_vocab_size, self.att_outer_size // num_heads)
        self.space_embed = nn.Embedding(5, self.att_outer_size)
        self.max_dep = 15
        self.lvl_embed = nn.Embedding(self.max_dep + 1, self.att_outer_size)
        self.word_embed.weight.data = torch.from_numpy(word_vectors.astype(np.float32))
        self.decoder = nn.LSTM(self.att_outer_size, rnn_hidden_size)
        self.leaf_encoder = nn.LSTM(self.att_outer_size, rnn_hidden_size)
        self.trg_key = nn.Linear(rnn_hidden_size * 1 + self.att_outer_size, self.att_outer_size, bias=False)
        #self.trg_key_fc1 = nn.Linear(rnn_hidden_size * 1 + self.att_outer_size, att_hidden_size, bias=False)
        #self.trg_key_fc2 = nn.Linear(att_hidden_size, self.att_outer_size, bias=False)
        self.layer_agg = nn.Linear(self.att_outer_size * (num_layers + 1), self.att_outer_size)

        self.hidden_fc1 = nn.Linear(self.att_outer_size, att_hidden_size, bias=False)
        self.hidden_fc2 = nn.Linear(self.att_hidden_size, self.rnn_hidden_size, bias=False)
        self.cell_fc1 = nn.Linear(self.att_outer_size, att_hidden_size, bias=False)
        self.cell_fc2 = nn.Linear(self.att_hidden_size, self.rnn_hidden_size, bias=False)

        self.self_attention_blocks = nn.ModuleList()
        self.ffn_blocks = nn.ModuleList()
        self.norm1_blocks = nn.ModuleList()
        self.norm2_blocks = nn.ModuleList()
        self.out_norm_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.self_attention_blocks.append(MultiheadAttention(self.att_outer_size,
                                                                 self.att_outer_size,
                                                                 self.att_outer_size,
                                                                 num_heads))
            self.ffn_blocks.append(FeedForwardNetwork(self.att_outer_size, att_hidden_size, dropout))
            self.norm1_blocks.append(nn.LayerNorm(self.att_outer_size))
            self.norm2_blocks.append(nn.LayerNorm(self.att_outer_size))
            self.out_norm_blocks.append(nn.LayerNorm(self.att_outer_size))
        self.out_norm = nn.LayerNorm(self.att_outer_size)

        self.dictionary = dictionary
        self.dropout = dropout

        for names in self.decoder._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.decoder, name)
                n = bias.size(0)
                start, end = n//4, n//2 
                bias.data[start:end].fill_(0.)

    def post_parameters(self):
        return []

    def flatten_parameters(self):
        self.decoder.flatten_parameters()

    def hidden_transform(self, hidden):
        return self.hidden_fc2(F.relu(self.hidden_fc1(hidden)))

    def cell_transform(self, cell):
        return self.cell_fc2(F.relu(self.cell_fc1(cell)))

    def init_hidden(self, src_hidden):
        hidden = self.hidden_transform(src_hidden)
        cell = self.cell_transform(src_hidden)
        return (hidden, cell)

    def forward(self, src_seqs, src_lengths, trg_seqs, trg_lengths, graph):
        batch_size = src_seqs.size(0)
        neigh_lst = [g[2] for g in graph]
        adj_lst = [g[0] for g in graph]
        prev_inds = [g[3] for g in graph]
        depths = [g[4] for g in graph]

        max_trg_len = max(trg_lengths)
        src_len = src_seqs.size(1)
        for x in range(batch_size):
            neigh_lst[x] = neigh_lst[x] + [[0]] * (max_trg_len - len(neigh_lst[x]))
            prev_inds[x] = prev_inds[x] + [0] * (max_trg_len - len(prev_inds[x]))
            depths[x] = depths[x] + [0] * (src_len - len(depths[x]))

        word_embed = self.word_embed_map(self.word_embed(src_seqs[:, :, 0].cuda()))
        depth_embed = torch.cuda.LongTensor(depths)
        depth_embed[depth_embed > self.max_dep] = self.max_dep
        depth_embed = self.lvl_embed(depth_embed)
        dep_embed = self.dep_embed(src_seqs[:, :, 2].cuda())
        pos_embed = self.pos_embed(src_seqs[:, :, 1].cuda())
        space_embed = self.space_embed(src_seqs[:, :, 4].cuda())
        
        word_embed = F.dropout(word_embed, p=self.dropout, training=self.training)
        depth_embed = F.dropout(depth_embed, p=self.dropout, training=self.training)
        dep_embed = F.dropout(dep_embed, p=self.dropout, training=self.training)
        pos_embed = F.dropout(pos_embed, p=self.dropout, training=self.training)
        space_embed = F.dropout(space_embed, p=self.dropout, training=self.training)
        
        '''
        encoder_input = torch.cat((word_embed,
                                   self.dep_embed(src_seqs[:, :, 2].cuda()),
                                   self.pos_embed(src_seqs[:, :, 1].cuda())),
                                  dim=2) + depth_embed
        '''
        encoder_input = word_embed + dep_embed + pos_embed + depth_embed + space_embed
        #encoder_input = word_embed + pos_embed + depth_embed + space_embed

        attention_mask = torch.arange(src_seqs.size(1)).expand(batch_size, src_seqs.size(1)) >= torch.LongTensor(src_lengths).expand(src_seqs.size(1), batch_size).transpose(0, 1)
        attention_bias = attention_mask.float() * -1e9
        attention_bias = attention_bias.cuda()

        neigh_mask = torch.ByteTensor(batch_size, src_len, src_len).fill_(0)
        dep_mask = torch.LongTensor(batch_size, src_len, src_len).fill_(0)
        for x in range(batch_size):
            for y in range(len(adj_lst[x])):
                neigh_mask[x, y, adj_lst[x][y]] = 1
                neigh_mask[x, adj_lst[x][y], y] = 1
                for z in range(len(adj_lst[x][y])):
                    dep_mask[x, y, z] = src_seqs[x, z, 2]
                    dep_mask[x, z, y] = src_seqs[x, z, 2]
        neigh_mask[:, torch.arange(src_len), torch.arange(src_len)] = 1 

        dep_key_bias = self.dep_key_embed(dep_mask.cuda())
        dep_val_bias = self.dep_val_embed(dep_mask.cuda())
        graph_bias = (~neigh_mask | attention_mask[:, None, :]).float() * -1e9
        graph_bias = graph_bias.cuda()

        #x = F.dropout(encoder_input, p=self.dropout, training=self.training)
        x = encoder_input
        layer_output = [F.dropout(self.out_norm(x), p=self.dropout, training=self.training)]
        #layer_output = [self.out_norm(x)]

        for self_attention, ffn, norm1, norm2 in \
            zip(self.self_attention_blocks,
                self.ffn_blocks,
                self.norm1_blocks,
                self.norm2_blocks):
            #y = self_attention(norm1(x), None, graph_bias)
            y = self_attention(norm1(x), None, graph_bias, None, None)
            #y = self_attention(norm1(x), None, graph_bias, dep_key_bias, dep_val_bias)
            x = residual(x, y, self.dropout, self.training)
            y = ffn(norm2(x))
            x = residual(x, y, self.dropout, self.training)
            layer_output.append(F.dropout(self.out_norm(x), p=self.dropout, training=self.training))
            #layer_output.append(self.out_norm(x))

        #encoder_outputs = self.out_norm(x)
        encoder_outputs = sum(layer_output)
        #layer_output = torch.cat(layer_output, dim=2).view(batch_size, src_len, -1).contiguous()
        #encoder_outputs = self.layer_agg(layer_output)
        #encoder_outputs = F.dropout(encoder_outputs, p=self.dropout, training=self.training)

        init_hidden = encoder_outputs.clone()
        init_hidden[attention_mask.unsqueeze(2).expand(batch_size, src_seqs.size(1), self.att_outer_size)] = 0
        init_hidden = init_hidden.sum(1).unsqueeze(0)
        decoder_hidden = self.init_hidden(init_hidden)

        #ans_embed = self.word_embed(src_seqs[:, :, 0].cuda())
        #ans_embed = F.dropout(ans_embed, p=self.dropout, training=self.training)
        ans_embed = F.dropout(encoder_outputs, p=self.dropout, training=self.training)
        #ans_embed = encoder_outputs
        trg_l = trg_seqs.size(1)

        decoder_input = torch.zeros_like(ans_embed[:, 0, :]).unsqueeze(0) 
        decoder_outputs = torch.FloatTensor(batch_size, trg_l, ans_embed.size(1)).cuda()
        
        for step in range(trg_l):
            if step:
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            else:
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                #decoder_output, decoder_hidden = self.decoder(decoder_input)
            decoder_output = F.dropout(decoder_output, p=self.dropout, training=self.training)
            step_neigh_lst = [lst[step] for lst in neigh_lst]
            #step_prev_inds = [lst[step] for lst in prev_inds]

            #neigh_output = torch.zeros(batch_size, self.att_outer_size).float().cuda()
            step_neigh_mask = torch.cuda.ByteTensor(batch_size, src_len).fill_(1)
            for x in range(batch_size):
                #neigh_output[x] = encoder_outputs[x, step_neigh_lst[x], :].sum(0)
                step_neigh_mask[x, step_neigh_lst[x]] = 0
            step_neigh_bias = step_neigh_mask.float() * -1e9
            neigh_output = self.neighbor_select(encoder_outputs, decoder_output, step_neigh_bias)
            
            #step_prev = leaf_hidden[torch.arange(batch_size), step_prev_inds].unsqueeze(0)
            decoder_output = torch.cat((decoder_output, neigh_output.unsqueeze(0)), dim=2)

            dot = self.attention_select(encoder_outputs, decoder_output.transpose(0, 1), step_neigh_bias)
            decoder_outputs[:, step, :] = F.log_softmax(dot.squeeze(1), dim=1)
            decoder_input = ans_embed[torch.arange(batch_size), trg_seqs[:, step], :].unsqueeze(0)

        return decoder_outputs

    def attention_select(self, encoder_outputs, decoder_output, attention_bias):
        decoder_output = self.trg_key(decoder_output)
        #decoder_output = self.trg_key_fc2(F.relu(self.trg_key_fc1(decoder_output)))
        dot = torch.bmm(decoder_output, encoder_outputs.transpose(1, 2))
        dot = dot + attention_bias[:, None, :]
        return dot

    def neighbor_select(self, encoder_outputs, decoder_output, step_neigh_bias):
        '''
        neigh_select_dot = torch.bmm(self.neigh_map(decoder_output).transpose(0, 1), encoder_outputs.transpose(1, 2))
        neigh_select_dot = neigh_select_dot + step_neigh_bias[:, None, :]
        #neigh_select_weight = F.softmax(neigh_select_dot, dim=2)
        neigh_select_weight = F.sigmoid(neigh_select_dot)
        '''
        neigh_select_weight = (step_neigh_bias == 0).float().unsqueeze(1)
        neigh_output = torch.bmm(neigh_select_weight, encoder_outputs).squeeze(1)
        return neigh_output

    def generate(self, src_seqs, src_lengths, beam_size, top_k, graph):
        '''
        batch_size = src_seqs.size(0)

        # embed tokens plus positions
        #encoder_input = self.word_embed(src_seqs[:, :, 0].cuda())
        encoder_input = torch.cat((self.word_embed(src_seqs[:, :, 0].cuda()),
                                   self.dep_embed(src_seqs[:, :, 2].cuda()),
                                   self.pos_embed(src_seqs[:, :, 1].cuda())),
                                  dim=2)
        attention_mask = torch.arange(src_seqs.size(1)).expand(batch_size, src_seqs.size(1)) >= torch.LongTensor(src_lengths).expand(src_seqs.size(1), batch_size).transpose(0, 1)
        attention_bias = attention_mask.float() * -1e9
        attention_bias = attention_bias.cuda()

        x = F.dropout(encoder_input, p=self.dropout, training=self.training)

        for self_attention, ffn, norm1, norm2 in zip(self.self_attention_blocks,
                                                     self.ffn_blocks,
                                                     self.norm1_blocks,
                                                     self.norm2_blocks):
            y = self_attention(norm1(x), None, attention_bias)
            x = residual(x, y, self.dropout, self.training)
            y = ffn(norm2(x))
            x = residual(x, y, self.dropout, self.training)
        encoder_outputs = self.out_norm(x)
        '''
        batch_size = src_seqs.size(0)
        adj_lst, root, neigh_lst, prev_inds, depths = graph[0]
        src_len = src_lengths[0]

        # embed tokens plus positions
        #encoder_input = self.word_embed(src_seqs[:, :, 0].cuda())
        word_embed = self.word_embed_map(self.word_embed(src_seqs[:, :, 0].cuda()))
        depth_embed = torch.cuda.LongTensor(depths)
        depth_embed[depth_embed > self.max_dep] = self.max_dep
        depth_embed = self.lvl_embed(depth_embed)
        dep_embed = self.dep_embed(src_seqs[:, :, 2].cuda())
        pos_embed = self.pos_embed(src_seqs[:, :, 1].cuda())
        space_embed = self.space_embed(src_seqs[:, :, 4].cuda())
        
        encoder_input = word_embed + dep_embed + pos_embed + depth_embed + space_embed
        attention_mask = torch.arange(src_seqs.size(1)).expand(batch_size, src_seqs.size(1)) >= torch.LongTensor(src_lengths).expand(src_seqs.size(1), batch_size).transpose(0, 1)
        attention_bias = attention_mask.float() * -1e9
        attention_bias = attention_bias.cuda()

        dep_mask = torch.LongTensor(batch_size, src_len, src_len).fill_(0)
        neigh_mask = torch.ByteTensor(batch_size, src_len, src_len).fill_(0)
        for y in range(len(adj_lst)):
            neigh_mask[0, y, adj_lst[y]] = 1
            neigh_mask[0, adj_lst[y], y] = 1
            for z in range(len(adj_lst[y])):
                dep_mask[0, y, z] = src_seqs[0, z, 2]
                dep_mask[0, z, y] = src_seqs[0, z, 2]
        neigh_mask[:, torch.arange(src_len), torch.arange(src_len)] = 1 

        dep_key_bias = self.dep_key_embed(dep_mask.cuda())
        dep_val_bias = self.dep_val_embed(dep_mask.cuda())
        graph_bias = (~neigh_mask | attention_mask[:, None, :]).float() * -1e9
        graph_bias = graph_bias.cuda()

        #x = F.dropout(encoder_input, p=self.dropout, training=self.training)
        x = encoder_input
        layer_output = [self.out_norm(x)]

        for self_attention, ffn, norm1, norm2 in \
            zip(self.self_attention_blocks,
                self.ffn_blocks,
                self.norm1_blocks,
                self.norm2_blocks):
            y = self_attention(norm1(x), None, graph_bias, None, None)
            #y = self_attention(norm1(x), None, graph_bias, dep_key_bias, dep_val_bias)
            x = residual(x, y, self.dropout, self.training)
            y = ffn(norm2(x))
            x = residual(x, y, self.dropout, self.training)
            layer_output.append(self.out_norm(x))

        #encoder_outputs = self.out_norm(x)
        encoder_outputs = sum(layer_output)

        init_hidden = encoder_outputs.clone()
        init_hidden[attention_mask.unsqueeze(2).expand(batch_size, src_seqs.size(1), self.att_outer_size)] = 0
        init_hidden = init_hidden.sum(1).unsqueeze(0)
        decoder_hidden = self.init_hidden(init_hidden)

        decoder_input = torch.zeros_like(encoder_outputs[:, 0, :]).unsqueeze(1) 
        src_length = encoder_outputs.size(1)
        beam_size = min(beam_size, src_length)

        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        step_neigh_bias = torch.ones(1, src_len).float().cuda() * -1e9
        neigh_output = self.neighbor_select(encoder_outputs, decoder_output, step_neigh_bias)
        decoder_output = torch.cat((decoder_output, neigh_output.unsqueeze(0)), dim=2)
        dot = self.attention_select(encoder_outputs, decoder_output, attention_bias)

        logits = F.log_softmax(dot.squeeze(1), dim=1)
        beam = torch.zeros(beam_size, src_length).long().cuda()
        beam_probs = torch.zeros(beam_size).float().cuda()
        logprobs, argtop = torch.topk(logits, beam_size, dim=1)
        argtop = argtop.squeeze(0)
        beam[:, 0] = argtop
        beam_probs = logprobs.clone().squeeze(0)
        
        decoder_hiddens = decoder_hidden[0].expand(1, beam_size, self.rnn_hidden_size).contiguous()
        decoder_cells = decoder_hidden[1].expand(1, beam_size, self.rnn_hidden_size).contiguous()
        decoder_input = encoder_outputs[0, argtop, :].unsqueeze(0)
        encoder_outputs = encoder_outputs.expand(beam_size, src_length, self.att_outer_size).contiguous()
        attention_bias = attention_bias.expand(beam_size, src_length).contiguous()
        
        mask = torch.cuda.ByteTensor(beam_size, src_length).fill_(0)
        mask[torch.arange(beam_size), beam[:, 0]] = 1
        for t in range(src_length - 1):
            top_k = min(top_k, src_length - t)

            decoder_output, decoder_hidden = self.decoder(decoder_input, (decoder_hiddens, decoder_cells))
            step_neigh_bias = mask.float() * -1e9
            neigh_output = self.neighbor_select(encoder_outputs, decoder_output, step_neigh_bias)
            decoder_output = torch.cat((decoder_output, neigh_output.unsqueeze(0)), dim=2)
            dot = self.attention_select(encoder_outputs, decoder_output.transpose(0, 1), attention_bias).squeeze(1)
            dot[mask] = -np.inf

            logits = F.log_softmax(dot, dim=1)
            logprobs, argtop = torch.topk(logits, top_k, dim=1)
            
            total_probs = beam_probs.unsqueeze(1).expand(beam_size, top_k).contiguous()
            total_probs = total_probs + logprobs
            best_probs, best_args = total_probs.view(-1).topk(beam_size)
            
            _decoder_hiddens = decoder_hiddens.clone()
            _decoder_cells = decoder_cells.clone()

            last = (best_args / top_k)
            curr = (best_args % top_k)
            beam = beam[last]
            beam_probs = best_probs
            beam[:, t+1] = argtop[last, curr]
            mask = mask[last]
            mask[torch.arange(beam_size), beam[:, t+1]] = 1
            
            decoder_hiddens = _decoder_hiddens[:, last, :]
            decoder_cells = _decoder_cells[:, last, :]
            decoder_input = encoder_outputs[0, beam[:, t+1], :].unsqueeze(0)
        
        best, best_arg = beam_probs.max(0)
        best_order = beam[best_arg]
        generations = src_seqs[0, best_order, 0]
        print(best.item() / src_length)
        return generations, best

    def tree_generate(self, src_seqs, src_lengths, beam_size, top_k, order, graph):
        batch_size = src_seqs.size(0)
        adj_lst, root, neigh_lst, prev_inds, depths, _ = graph
        src_len = src_lengths[0]

        # embed tokens plus positions
        #encoder_input = self.word_embed(src_seqs[:, :, 0].cuda())
        word_embed = self.word_embed_map(self.word_embed(src_seqs[:, :, 0].cuda()))
        depth_embed = torch.cuda.LongTensor(depths)
        depth_embed[depth_embed > self.max_dep] = self.max_dep
        depth_embed = self.lvl_embed(depth_embed)
        dep_embed = self.dep_embed(src_seqs[:, :, 2].cuda())
        pos_embed = self.pos_embed(src_seqs[:, :, 1].cuda())
        space_embed = self.space_embed(src_seqs[:, :, 4].cuda())
        
        '''
        encoder_input = torch.cat((word_embed,
                                   self.dep_embed(src_seqs[:, :, 2].cuda()),
                                   self.pos_embed(src_seqs[:, :, 1].cuda())),
                                  dim=2) + depth_embed
        '''
        encoder_input = word_embed + dep_embed + pos_embed + depth_embed + space_embed
        attention_mask = torch.arange(src_seqs.size(1)).expand(batch_size, src_seqs.size(1)) >= torch.LongTensor(src_lengths).expand(src_seqs.size(1), batch_size).transpose(0, 1)
        attention_bias = attention_mask.float() * -1e9
        attention_bias = attention_bias.cuda()

        dep_mask = torch.LongTensor(batch_size, src_len, src_len).fill_(0)
        neigh_mask = torch.ByteTensor(batch_size, src_len, src_len).fill_(0)
        for y in range(len(adj_lst)):
            neigh_mask[0, y, adj_lst[y]] = 1
            neigh_mask[0, adj_lst[y], y] = 1
            for z in range(len(adj_lst[y])):
                dep_mask[0, y, z] = src_seqs[0, z, 2]
                dep_mask[0, z, y] = src_seqs[0, z, 2]
        neigh_mask[:, torch.arange(src_len), torch.arange(src_len)] = 1 

        dep_key_bias = self.dep_key_embed(dep_mask.cuda())
        dep_val_bias = self.dep_val_embed(dep_mask.cuda())
        graph_bias = (~neigh_mask | attention_mask[:, None, :]).float() * -1e9
        graph_bias = graph_bias.cuda()

        #x = F.dropout(encoder_input, p=self.dropout, training=self.training)
        x = encoder_input
        layer_output = [self.out_norm(x)]

        for self_attention, ffn, norm1, norm2 in \
            zip(self.self_attention_blocks,
                self.ffn_blocks,
                self.norm1_blocks,
                self.norm2_blocks):
            y = self_attention(norm1(x), None, graph_bias, None, None)
            #y = self_attention(norm1(x), None, graph_bias, dep_key_bias, dep_val_bias)
            x = residual(x, y, self.dropout, self.training)
            y = ffn(norm2(x))
            x = residual(x, y, self.dropout, self.training)
            layer_output.append(self.out_norm(x))

        #encoder_outputs = self.out_norm(x)
        encoder_outputs = sum(layer_output)

        init_hidden = encoder_outputs.clone()
        init_hidden[attention_mask.unsqueeze(2).expand(batch_size, src_seqs.size(1), self.att_outer_size)] = 0
        init_hidden = init_hidden.sum(1).unsqueeze(0)
        decoder_hidden = self.init_hidden(init_hidden)

        decoder_input = torch.zeros_like(encoder_outputs[:, 0, :]).unsqueeze(1) 
        src_length = encoder_outputs.size(1)
 
        '''
        leaf_encoder_input = torch.zeros(1, 1, self.att_outer_size).cuda()
        leaf_output, leaf_hidden = self.leaf_encoder(leaf_encoder_input)
        '''

        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        step_neigh_mask = torch.cuda.ByteTensor(1, src_len).fill_(1)
        step_neigh_mask[0, neigh_lst[0]] = 0
        step_neigh_bias = step_neigh_mask.float() * -1e9
        neigh_output = self.neighbor_select(encoder_outputs, decoder_output, step_neigh_bias)

        decoder_output = torch.cat((decoder_output, neigh_output.unsqueeze(0)), dim=2)
        dot = self.attention_select(encoder_outputs, decoder_output, step_neigh_bias).squeeze(1)
        par = root
        children = adj_lst[par] + [par]
        mask = [y for y in range(src_length) if y not in children]
        dot[:, mask] = -np.inf

        non_leaf = max(1, sum(1 for x in adj_lst if len(x) > 0))
        top_k = min(top_k, src_length)
        if order == 'hybrid':
            logits = F.log_softmax(dot, dim=1)
            logprobs, argtop = torch.topk(logits, top_k, dim=1)
            beam = torch.zeros(beam_size, src_length + non_leaf - 1).long().cuda()
            beam_probs = torch.zeros(beam_size).float().cuda()
            beam_probs[:] = -np.inf
            beam[:top_k, 0] = argtop.squeeze(0)
            beam_probs[:top_k] = logprobs[0]
            decode_len = src_length + non_leaf - 1 - 1 
        elif order == 'depth':
            logits = F.log_softmax(dot, dim=1)
            beam = torch.zeros(beam_size, src_length + non_leaf).long().cuda()
            beam_probs = torch.zeros(beam_size).float().cuda()
            beam_probs[:] = -np.inf
            beam[0, 0] = root
            beam_probs[0] = logits[0, root]
            decode_len = src_length + non_leaf - 1

        decoder_hiddens = decoder_hidden[0].expand(1, beam_size, self.rnn_hidden_size).contiguous()
        decoder_cells = decoder_hidden[1].expand(1, beam_size, self.rnn_hidden_size).contiguous()
        #leaf_decoder_hiddens = leaf_hidden[0].expand(1, beam_size, self.rnn_hidden_size).contiguous()
        #leaf_decoder_cells = leaf_hidden[1].expand(1, beam_size, self.rnn_hidden_size).contiguous()
        decoder_input = encoder_outputs[0, [root] * beam_size, :].unsqueeze(0)
        encoder_outputs = encoder_outputs.expand(beam_size, src_length, self.att_outer_size).contiguous()
        attention_bias = attention_bias.expand(beam_size, src_length).contiguous()
        
        current = np.array([Node(root) for x in range(beam_size)])
        if order == 'hybrid':
            for x in range(beam_size):
                node = Node(beam[x, 0].item())
                node.parent = current[x]

        for t in range(decode_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, (decoder_hiddens, decoder_cells))

            neigh_output = torch.zeros(beam_size, self.att_outer_size).float().cuda()
            mask = torch.cuda.ByteTensor(beam_size, src_length).fill_(0)
            step_neigh_mask = torch.cuda.ByteTensor(beam_size, src_length).fill_(1)
            for x in range(beam_size):
                if order == 'hybrid':
                    par = current[x].name
                    prev = [ch.name for ch in current[x].children]
                    children = adj_lst[par] + [par]
                    if len(prev) == len(children):
                        i = 0
                        while i < len(children):
                            if prev[i] != par and adj_lst[prev[i]]:
                                current[x] = current[x].children[i]
                                break
                            else:
                                i += 1
                        if i == len(children):
                            while current[x].parent is not None:
                                curr_name = current[x].name
                                current[x] = current[x].parent
                                names = [ch.name for ch in current[x].children]
                                ind = names.index(curr_name)
                                ind += 1
                                found = False
                                while ind < len(names):
                                    if names[ind] != current[x].name and adj_lst[names[ind]]:
                                        found = True
                                        current[x] = current[x].children[ind]
                                        break
                                    ind += 1
                                if found:
                                    break
                        par = current[x].name
                        prev = [ch.name for ch in current[x].children]
                        children = adj_lst[par] + [par]
                        _mask = [y for y in range(src_length) if y not in children]
                    else:
                        _mask = [y for y in range(src_length) if y not in children or y in prev]
                    mask[x, _mask] = 1
                    neigh_select = [ch for ch in children if ch not in prev]
                    #neigh_output[x] = encoder_outputs[x, neigh_select, :].sum(0)
                    step_neigh_mask[x, neigh_select] = 0
         
                elif order == 'depth':
                    par = current[x].name
                    prev = [ch.name for ch in current[x].children]
                    children = adj_lst[par] + [par]
                    if len(children) == 1 or (current[x].parent is not None and par == current[x].parent.name):
                        while current[x].parent is not None:
                            current[x] = current[x].parent
                            par = current[x].name
                            children = adj_lst[par] + [par]
                            prev = [ch.name for ch in current[x].children]
                            if len(prev) < len(children):
                                break
                    _mask = [y for y in range(src_length) if y not in children or y in prev]
                    mask[x, _mask] = 1
                    neigh_select = [ch for ch in children if ch not in prev]
                    neigh_output[x] = encoder_outputs[x, neigh_select, :].sum(0)

            step_neigh_bias = step_neigh_mask.float() * -1e9
            neigh_output = self.neighbor_select(encoder_outputs, decoder_output, step_neigh_bias)

            decoder_output = torch.cat((decoder_output, neigh_output.unsqueeze(0)), dim=2)
            dot = self.attention_select(encoder_outputs, decoder_output.transpose(0, 1), step_neigh_bias).squeeze(1)
            dot[mask] = -np.inf

            logits = F.log_softmax(dot, dim=1)
            logprobs, argtop = torch.topk(logits, top_k, dim=1)
            
            total_probs = beam_probs.unsqueeze(1).expand(beam_size, top_k).contiguous()
            total_probs = total_probs + logprobs
            best_probs, best_args = total_probs.view(-1).topk(beam_size)
            
            _decoder_hiddens = decoder_hiddens.clone()
            _decoder_cells = decoder_cells.clone()
            #_leaf_decoder_hiddens = leaf_decoder_hiddens.clone()
            #_leaf_decoder_cells = leaf_decoder_cells.clone()

            last = (best_args / top_k)
            curr = (best_args % top_k)
            beam = beam[last]
            beam_probs = best_probs
            beam[:, t+1] = argtop[last, curr]
            
            decoder_hiddens = _decoder_hiddens[:, last, :]
            decoder_cells = _decoder_cells[:, last, :]
            #leaf_decoder_hiddens = _leaf_decoder_hiddens[:, last, :]
            #leaf_decoder_cells = _leaf_decoder_cells[:, last, :]
            decoder_input = encoder_outputs[0, beam[:, t+1], :].unsqueeze(0)

            new_curr = []
            for x in range(beam_size):
                curr = deepcopy(current[last[x].item()])
                node = Node(beam[x, t+1].item())
                node.parent = curr
                if order == 'hybrid':
                    new_curr.append(curr)
                elif order == 'depth':
                    new_curr.append(node)
            current = new_curr

        best, best_arg = beam_probs.max(0)
        best_order = beam[best_arg]
        best_tree = current[best_arg]
        while best_tree.parent is not None:
            best_tree = best_tree.parent

        def agg(tree):
            if not tree.children:
                return [tree.name]
            else:
                return list(itertools.chain.from_iterable([agg(ch) for ch in tree.children]))

        best_order = agg(best_tree)
        generations = src_seqs[0, best_order, 0]
        print(best.item() / src_length)
        return generations, best, best_order


    def masked_loss(self, logits, target, mask):
        batch_size = logits.size(0)
        log_probs_flat = F.log_softmax(logits, dim=2).view(-1, logits.size(-1))
        target_flat = target.contiguous().view(-1, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        losses = losses_flat.view(*target.size())
        losses = losses * mask.float()
        loss = losses.sum() / mask.float().sum()
        return loss, mask.float().sum()

    def loss(self, src_seqs, src_lengths, trg_seqs, trg_lengths, graph):
        batch_size = src_seqs.size(0)
        decoder_outputs = self.forward(src_seqs, src_lengths, trg_seqs, trg_lengths, graph)
        mask = torch.arange(trg_seqs.size(1)).expand(batch_size, trg_seqs.size(1)) < torch.LongTensor(trg_lengths).expand(trg_seqs.size(1), batch_size).transpose(0, 1)
        loss, count = self.masked_loss(decoder_outputs, trg_seqs.cuda(), mask.cuda())
        return loss
        


class T2_Transformer(nn.Module):
    def __init__(self, input_size, rnn_hidden_size, att_hidden_size, word_vectors, dictionary, dropout, num_layers, num_heads):
        super(T2_Transformer, self).__init__()
        self.rnn_hidden_size = rnn_hidden_size
        self.att_outer_size = 512
        self.att_hidden_size = att_hidden_size
        self.input_size = input_size
        self.word_vocab_size = len(dictionary['word']) + 1
        self.pos_vocab_size = len(dictionary['pos']) + 1
        self.dep_vocab_size = len(dictionary['dep']) + 1
        self.trg_word_vocab_size = len(dictionary['trg_word']) + 1
        self.trg_pos_vocab_size = len(dictionary['trg_pos']) + 1
        self.word_embed = nn.Embedding(self.word_vocab_size, input_size)
        self.word_embed_map = nn.Linear(input_size, self.att_outer_size)
        self.pos_embed = nn.Embedding(self.pos_vocab_size, self.att_outer_size)
        self.dep_embed = nn.Embedding(self.dep_vocab_size, self.att_outer_size)
        self.max_dep = 15
        self.lvl_embed = nn.Embedding(self.max_dep + 1, self.att_outer_size)
        self.word_embed.weight.data = torch.from_numpy(word_vectors.astype(np.float32))
        self.decoder = nn.LSTM(self.att_outer_size, rnn_hidden_size)
        self.leaf_encoder = nn.LSTM(self.att_outer_size, rnn_hidden_size)
        self.trg_key = nn.Linear(rnn_hidden_size * 1 + self.att_outer_size, self.att_outer_size, bias=False)
        #self.trg_key_fc1 = nn.Linear(rnn_hidden_size * 1 + self.att_outer_size, att_hidden_size, bias=False)
        #self.trg_key_fc2 = nn.Linear(att_hidden_size, self.att_outer_size, bias=False)
        self.par_pos = nn.Linear(self.att_outer_size * (num_layers + 1), self.trg_word_vocab_size)
        self.par_neg = nn.Linear(self.att_outer_size * (num_layers + 1), self.trg_word_vocab_size)

        self.hidden_fc1 = nn.Linear(self.att_outer_size, att_hidden_size, bias=False)
        self.hidden_fc2 = nn.Linear(self.att_hidden_size, self.rnn_hidden_size, bias=False)
        self.cell_fc1 = nn.Linear(self.att_outer_size, att_hidden_size, bias=False)
        self.cell_fc2 = nn.Linear(self.att_hidden_size, self.rnn_hidden_size, bias=False)

        self.self_attention_blocks = nn.ModuleList()
        self.ffn_blocks = nn.ModuleList()
        self.norm1_blocks = nn.ModuleList()
        self.norm2_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.self_attention_blocks.append(MultiheadAttention(self.att_outer_size,
                                                                 self.att_outer_size,
                                                                 self.att_outer_size,
                                                                 num_heads))
            self.ffn_blocks.append(FeedForwardNetwork(self.att_outer_size, att_hidden_size, dropout))
            self.norm1_blocks.append(nn.LayerNorm(self.att_outer_size))
            self.norm2_blocks.append(nn.LayerNorm(self.att_outer_size))
        self.out_norm = nn.LayerNorm(self.att_outer_size)

        self.dictionary = dictionary
        self.dropout = dropout

        for names in self.decoder._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.decoder, name)
                n = bias.size(0)
                start, end = n//4, n//2 
                bias.data[start:end].fill_(0.)

    def post_parameters(self):
        return []

    def flatten_parameters(self):
        self.decoder.flatten_parameters()

    def hidden_transform(self, hidden):
        return self.hidden_fc2(F.relu(self.hidden_fc1(hidden)))

    def cell_transform(self, cell):
        return self.cell_fc2(F.relu(self.cell_fc1(cell)))

    def init_hidden(self, src_hidden):
        hidden = self.hidden_transform(src_hidden)
        cell = self.cell_transform(src_hidden)
        return (hidden, cell)

    def forward(self, src_seqs, src_lengths, trg_seqs, trg_lengths, graph):
        batch_size = src_seqs.size(0)
        trg_lst = [g[2] for g in graph]
        adj_lst = [g[0] for g in graph]
        depths = [g[3] for g in graph]

        max_trg_len = max(trg_lengths)
        src_len = src_seqs.size(1)
        for x in range(batch_size):
            #trg_lst[x] = trg_lst[x] + [[0]] * (max_trg_len - len(trg_lst[x]))
            depths[x] = depths[x] + [0] * (src_len - len(depths[x]))

        word_embed = self.word_embed_map(self.word_embed(src_seqs[:, :, 0].cuda()))
        depth_embed = torch.cuda.LongTensor(depths)
        depth_embed[depth_embed > self.max_dep] = self.max_dep
        depth_embed = self.lvl_embed(depth_embed)
        dep_embed = self.dep_embed(src_seqs[:, :, 2].cuda())
        pos_embed = self.pos_embed(src_seqs[:, :, 1].cuda())
        
        word_embed = F.dropout(word_embed, p=self.dropout, training=self.training)
        depth_embed = F.dropout(depth_embed, p=self.dropout, training=self.training)
        dep_embed = F.dropout(dep_embed, p=self.dropout, training=self.training)
        pos_embed = F.dropout(pos_embed, p=self.dropout, training=self.training)
        
        '''
        encoder_input = torch.cat((word_embed,
                                   self.dep_embed(src_seqs[:, :, 2].cuda()),
                                   self.pos_embed(src_seqs[:, :, 1].cuda())),
                                  dim=2) + depth_embed
        '''
        encoder_input = word_embed + dep_embed + pos_embed + depth_embed

        attention_mask = torch.arange(src_seqs.size(1)).expand(batch_size, src_seqs.size(1)) >= torch.LongTensor(src_lengths).expand(src_seqs.size(1), batch_size).transpose(0, 1)
        attention_bias = attention_mask.float() * -1e9
        attention_bias = attention_bias.cuda()

        neigh_mask = torch.ByteTensor(batch_size, src_len, src_len).fill_(0)
        for x in range(batch_size):
            for y in range(len(adj_lst[x])):
                neigh_mask[x, y, adj_lst[x][y]] = 1
                neigh_mask[x, adj_lst[x][y], y] = 1
        neigh_mask[:, torch.arange(src_len), torch.arange(src_len)] = 1 

        graph_bias = (~neigh_mask | attention_mask[:, None, :]).float() * -1e9
        graph_bias = graph_bias.cuda()

        #x = F.dropout(encoder_input, p=self.dropout, training=self.training)
        x = encoder_input
        layer_output = [self.out_norm(x).unsqueeze(2)]

        for self_attention, ffn, norm1, norm2 in zip(self.self_attention_blocks,
                                                     self.ffn_blocks,
                                                     self.norm1_blocks,
                                                     self.norm2_blocks):
            #y = self_attention(norm1(x), None, attention_bias)
            y = self_attention(norm1(x), None, graph_bias, None, None)
            x = residual(x, y, self.dropout, self.training)
            y = ffn(norm2(x))
            x = residual(x, y, self.dropout, self.training)
            layer_output.append(self.out_norm(x).unsqueeze(2))

        #encoder_outputs = F.dropout(self.out_norm(x), self.dropout, self.training)
        encoder_outputs = torch.cat(layer_output, dim=2).view(batch_size, src_len, -1).contiguous()
        encoder_outputs = F.dropout(encoder_outputs, self.dropout, self.training)

        x_ind = []
        y_ind = []
        for x in range(batch_size): 
            y_ind.extend(list(range(src_lengths[x])))
            x_ind.extend([x] * src_lengths[x])
        par_encoding = encoder_outputs[x_ind, y_ind, :]
        par_pos = self.par_pos(par_encoding).unsqueeze(2)
        par_neg = self.par_neg(par_encoding).unsqueeze(2)
        par_out = torch.cat((par_neg, par_pos), dim=2)
        par_out = F.log_softmax(par_out, dim=2)
        labels = torch.cuda.LongTensor(len(x_ind), self.trg_word_vocab_size).fill_(0)

        offset = 0
        for x in range(batch_size):
            for trg in trg_lst[x]:
                labels[offset+trg[2], trg[0]] = 1
            offset += src_lengths[x]

        return par_out.view(-1, 2).contiguous(), labels.view(-1).contiguous()

    def attention_select(self, encoder_outputs, decoder_output, attention_bias):
        decoder_output = self.trg_key(decoder_output)
        #decoder_output = self.trg_key_fc2(F.relu(self.trg_key_fc1(decoder_output)))
        dot = torch.bmm(decoder_output, encoder_outputs.transpose(1, 2))
        dot = dot + attention_bias[:, None, :]
        return dot

    def neighbor_select(self, encoder_outputs, decoder_output, step_neigh_bias):
        '''
        neigh_select_dot = torch.bmm(self.neigh_map(decoder_output).transpose(0, 1), encoder_outputs.transpose(1, 2))
        neigh_select_dot = neigh_select_dot + step_neigh_bias[:, None, :]
        #neigh_select_weight = F.softmax(neigh_select_dot, dim=2)
        neigh_select_weight = F.sigmoid(neigh_select_dot)
        '''
        neigh_select_weight = (step_neigh_bias == 0).float().unsqueeze(1)
        neigh_output = torch.bmm(neigh_select_weight, encoder_outputs).squeeze(1)
        return neigh_output

    def generate(self, src_seqs, src_lengths, beam_size, top_k, order, graph):
        batch_size = src_seqs.size(0)
        adj_lst, root, trg_lst, depths = graph
        src_len = src_lengths[0]

        word_embed = self.word_embed_map(self.word_embed(src_seqs[:, :, 0].cuda()))
        depth_embed = torch.cuda.LongTensor(depths)
        depth_embed[depth_embed > self.max_dep] = self.max_dep
        depth_embed = self.lvl_embed(depth_embed)
        dep_embed = self.dep_embed(src_seqs[:, :, 2].cuda())
        pos_embed = self.pos_embed(src_seqs[:, :, 1].cuda())
        
        '''
        encoder_input = torch.cat((self.word_embed(src_seqs[:, :, 0].cuda()),
                                   self.dep_embed(src_seqs[:, :, 2].cuda()),
                                   self.pos_embed(src_seqs[:, :, 1].cuda())),
                                  dim=2)
        '''
        encoder_input = word_embed + depth_embed + dep_embed + pos_embed

        attention_mask = torch.arange(src_seqs.size(1)).expand(batch_size, src_seqs.size(1)) >= torch.LongTensor(src_lengths).expand(src_seqs.size(1), batch_size).transpose(0, 1)
        attention_bias = attention_mask.float() * -1e9
        attention_bias = attention_bias.cuda()

        neigh_mask = torch.ByteTensor(batch_size, src_len, src_len).fill_(0)
        for y in range(len(adj_lst)):
            neigh_mask[0, y, adj_lst[y]] = 1
            neigh_mask[0, adj_lst[y], y] = 1
        neigh_mask[:, torch.arange(src_len), torch.arange(src_len)] = 1 

        graph_bias = (~neigh_mask | attention_mask[:, None, :]).float() * -1e9
        graph_bias = graph_bias.cuda()

        x = encoder_input
        layer_output = [self.out_norm(x).unsqueeze(2)]

        for self_attention, ffn, norm1, norm2 in zip(self.self_attention_blocks,
                                                     self.ffn_blocks,
                                                     self.norm1_blocks,
                                                     self.norm2_blocks):
            y = self_attention(norm1(x), None, graph_bias, None, None)
            x = residual(x, y, self.dropout, self.training)
            y = ffn(norm2(x))
            x = residual(x, y, self.dropout, self.training)
            layer_output.append(self.out_norm(x).unsqueeze(2))

        #encoder_outputs = self.out_norm(x)
        encoder_outputs = torch.cat(layer_output, dim=2).view(batch_size, src_len, -1).contiguous()
            
        x_ind = []
        y_ind = []
        for x in range(batch_size): 
            y_ind.extend(list(range(src_lengths[x])))
            x_ind.extend([x] * src_lengths[x])
        par_encoding = encoder_outputs[x_ind, y_ind, :]
        par_pos = self.par_pos(par_encoding).unsqueeze(2)
        par_neg = self.par_neg(par_encoding).unsqueeze(2)
        par_out = torch.cat((par_neg, par_pos), dim=2)
        par_out = F.softmax(par_out, dim=2)
        words = (par_out[:, :, 1] > 0.5).nonzero()

        return words.tolist()

    def masked_loss(self, logits, target, mask):
        batch_size = logits.size(0)
        log_probs_flat = F.log_softmax(logits, dim=2).view(-1, logits.size(-1))
        target_flat = target.contiguous().view(-1, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        losses = losses_flat.view(*target.size())
        losses = losses * mask.float()
        loss = losses.sum() / mask.float().sum()
        return loss, mask.float().sum()

    def loss(self, src_seqs, src_lengths, trg_seqs, trg_lengths, graph):
        batch_size = src_seqs.size(0)
        logits, labels = self.forward(src_seqs, src_lengths, trg_seqs, trg_lengths, graph)
        pred = logits.max(1)[1]
        pos_num = (labels == 1).long().sum()
        neg_num = (labels == 0).long().sum()
        pos_loss = logits[labels == 1, 1].sum() / pos_num.float()
        neg_loss = logits[labels == 0, 0].sum() / neg_num.float()
        true_pos = ((pred & labels) == 1).sum().float()
        false = (pred != labels).sum().float()
        f1 = 2 * true_pos / (2 * true_pos + false)
        #print (logits[:, 1] > logits[:, 0]).float().sum().item()
        #return - pos_loss - 200 * neg_loss, f1
        return -logits[torch.arange(labels.size(0)), labels].sum() / 100, f1

        
class T5_Transformer(nn.Module):
    def __init__(self, input_size, rnn_hidden_size, att_hidden_size, word_vectors, dictionary, dropout, num_layers, num_heads):
        super(T5_Transformer, self).__init__()
        self.rnn_hidden_size = rnn_hidden_size
        self.att_outer_size = 512
        self.att_hidden_size = att_hidden_size
        self.input_size = input_size
        self.word_vocab_size = len(dictionary['word']) + 1
        self.pos_vocab_size = len(dictionary['pos']) + 1
        self.dep_vocab_size = len(dictionary['dep']) + 1
        self.trg_word_vocab_size = len(dictionary['trg_word']) + 1
        self.word_embed = nn.Embedding(self.word_vocab_size, input_size)
        self.trg_word_embed = nn.Embedding(self.trg_word_vocab_size, self.att_outer_size)
        self.word_embed_map = nn.Linear(input_size, self.att_outer_size)
        self.choose = nn.Sequential(
                        nn.Linear(rnn_hidden_size, rnn_hidden_size),
                        nn.ReLU(),
                        nn.Linear(rnn_hidden_size, 2))
        self.trg_word_map = nn.Sequential(
                                nn.Linear(rnn_hidden_size, rnn_hidden_size),
                                nn.Tanh(),
                                nn.Linear(self.att_outer_size * 0 + rnn_hidden_size, self.trg_word_vocab_size))
        self.pos_embed = nn.Embedding(self.pos_vocab_size, self.att_outer_size)
        self.dep_embed = nn.Embedding(self.dep_vocab_size, self.att_outer_size)
        self.space_embed = nn.Embedding(5, self.att_outer_size)
        self.max_dep = 15
        self.lvl_embed = nn.Embedding(self.max_dep + 1, self.att_outer_size)
        self.word_embed.weight.data = torch.from_numpy(word_vectors.astype(np.float32))
        self.decoder = nn.LSTM(self.att_outer_size, rnn_hidden_size)
        self.func_decoder = nn.LSTM(self.att_outer_size, rnn_hidden_size)
        self.trg_key = nn.Linear(rnn_hidden_size * 1 + self.att_outer_size, self.att_outer_size, bias=False)
        #self.trg_key_fc1 = nn.Linear(rnn_hidden_size * 1 + self.att_outer_size, att_hidden_size, bias=False)
        #self.trg_key_fc2 = nn.Linear(att_hidden_size, self.att_outer_size, bias=False)
        self.layer_agg = nn.Linear(self.att_outer_size * (num_layers + 1), self.att_outer_size)

        self.hidden_fc1 = nn.Linear(self.att_outer_size, att_hidden_size, bias=False)
        self.hidden_fc2 = nn.Linear(self.att_hidden_size, self.rnn_hidden_size, bias=False)
        self.cell_fc1 = nn.Linear(self.att_outer_size, att_hidden_size, bias=False)
        self.cell_fc2 = nn.Linear(self.att_hidden_size, self.rnn_hidden_size, bias=False)

        self.self_attention_blocks = nn.ModuleList()
        self.ffn_blocks = nn.ModuleList()
        self.norm1_blocks = nn.ModuleList()
        self.norm2_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.self_attention_blocks.append(MultiheadAttention(self.att_outer_size,
                                                                 self.att_outer_size,
                                                                 self.att_outer_size,
                                                                 num_heads))
            self.ffn_blocks.append(FeedForwardNetwork(self.att_outer_size, att_hidden_size, dropout))
            self.norm1_blocks.append(nn.LayerNorm(self.att_outer_size))
            self.norm2_blocks.append(nn.LayerNorm(self.att_outer_size))
        self.out_norm = nn.LayerNorm(self.att_outer_size)

        self.dictionary = dictionary
        self.dropout = dropout

        for names in self.decoder._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.decoder, name)
                n = bias.size(0)
                start, end = n//4, n//2 
                bias.data[start:end].fill_(0.)

    def post_parameters(self):
        return []

    def flatten_parameters(self):
        self.decoder.flatten_parameters()

    def hidden_transform(self, hidden):
        return self.hidden_fc2(F.relu(self.hidden_fc1(hidden)))

    def cell_transform(self, cell):
        return self.cell_fc2(F.relu(self.cell_fc1(cell)))

    def init_hidden(self, src_hidden):
        hidden = self.hidden_transform(src_hidden)
        cell = self.cell_transform(src_hidden)
        return (hidden, cell)

    def forward(self, src_seqs, src_lengths, trg_seqs, trg_lengths, graph):
        batch_size = src_seqs.size(0)
        neigh_lst = [g[2] for g in graph]
        adj_lst = [g[0] for g in graph]
        par_inds = [g[3] for g in graph]
        depths = [g[4] for g in graph]
        func_mask = [g[5] for g in graph]

        max_trg_len = max(trg_lengths)
        src_len = src_seqs.size(1)
        for x in range(batch_size):
            neigh_lst[x] = neigh_lst[x] + [[0]] * (max_trg_len - len(neigh_lst[x]))
            par_inds[x] = par_inds[x] + [0] * (max_trg_len - len(par_inds[x]))
            depths[x] = depths[x] + [0] * (src_len - len(depths[x]))
            func_mask[x] = func_mask[x] + [0] * (max_trg_len - len(func_mask[x]))

        word_embed = self.word_embed_map(self.word_embed(src_seqs[:, :, 0].cuda()))
        depth_embed = torch.cuda.LongTensor(depths)
        depth_embed[depth_embed > self.max_dep] = self.max_dep
        depth_embed = self.lvl_embed(depth_embed)
        dep_embed = self.dep_embed(src_seqs[:, :, 2].cuda())
        pos_embed = self.pos_embed(src_seqs[:, :, 1].cuda())
        space_embed = self.space_embed(src_seqs[:, :, 4].cuda())
        
        word_embed = F.dropout(word_embed, p=self.dropout, training=self.training)
        depth_embed = F.dropout(depth_embed, p=self.dropout, training=self.training)
        dep_embed = F.dropout(dep_embed, p=self.dropout, training=self.training)
        pos_embed = F.dropout(pos_embed, p=self.dropout, training=self.training)
        space_embed = F.dropout(space_embed, p=self.dropout, training=self.training)
        
        encoder_input = word_embed + dep_embed + pos_embed + depth_embed + space_embed

        attention_mask = torch.arange(src_seqs.size(1)).expand(batch_size, src_seqs.size(1)) >= torch.LongTensor(src_lengths).expand(src_seqs.size(1), batch_size).transpose(0, 1)
        attention_bias = attention_mask.float() * -1e9
        attention_bias = attention_bias.cuda()

        neigh_mask = torch.ByteTensor(batch_size, src_len, src_len).fill_(0)
        for x in range(batch_size):
            for y in range(len(adj_lst[x])):
                neigh_mask[x, y, adj_lst[x][y]] = 1
                neigh_mask[x, adj_lst[x][y], y] = 1
        neigh_mask[:, torch.arange(src_len), torch.arange(src_len)] = 1 

        graph_bias = (~neigh_mask | attention_mask[:, None, :]).float() * -1e9
        graph_bias = graph_bias.cuda()

        func_mask = torch.cuda.FloatTensor(func_mask)
        par_inds = torch.cuda.LongTensor(par_inds)

        #x = F.dropout(encoder_input, p=self.dropout, training=self.training)
        x = encoder_input
        layer_output = [F.dropout(self.out_norm(x), p=self.dropout, training=self.training)]
        #layer_output = [self.out_norm(x)]

        for self_attention, ffn, norm1, norm2 in zip(self.self_attention_blocks,
                                                     self.ffn_blocks,
                                                     self.norm1_blocks,
                                                     self.norm2_blocks):
            #y = self_attention(norm1(x), None, attention_bias)
            y = self_attention(norm1(x), None, graph_bias, None, None)
            x = residual(x, y, self.dropout, self.training)
            y = ffn(norm2(x))
            x = residual(x, y, self.dropout, self.training)
            layer_output.append(F.dropout(self.out_norm(x), p=self.dropout, training=self.training))
            #layer_output.append(self.out_norm(x))

        #encoder_outputs = self.out_norm(x)
        encoder_outputs = sum(layer_output)
        #layer_output = torch.cat(layer_output, dim=2).view(batch_size, src_len, -1).contiguous()
        #encoder_outputs = self.layer_agg(layer_output)
        #encoder_outputs = F.dropout(encoder_outputs, p=self.dropout, training=self.training)

        init_hidden = encoder_outputs.clone()
        init_hidden[attention_mask.unsqueeze(2).expand(batch_size, src_seqs.size(1), self.att_outer_size)] = 0
        init_hidden = init_hidden.sum(1).unsqueeze(0)
        decoder_hidden = self.init_hidden(init_hidden)

        #ans_embed = self.word_embed(src_seqs[:, :, 0].cuda())
        #ans_embed = F.dropout(ans_embed, p=self.dropout, training=self.training)
        ans_embed = F.dropout(encoder_outputs, p=self.dropout, training=self.training)
        #ans_embed = encoder_outputs
        trg_l = trg_seqs.size(1)

        decoder_input = torch.zeros_like(ans_embed[:, 0, :]).unsqueeze(0) 
        func_decoder_input = torch.zeros_like(ans_embed[:, 0, :]).unsqueeze(0) 
        decoder_outputs = torch.FloatTensor(batch_size, trg_l, ans_embed.size(1) + self.trg_word_vocab_size).cuda()
        choose_probs = torch.zeros(batch_size, trg_l, 2).cuda()
        
        for step in range(trg_l):
            if step:
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                #func_decoder_output, func_decoder_hidden = self.func_decoder(func_decoder_input, func_decoder_hidden)
            else:
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                #func_decoder_output, func_decoder_hidden = self.func_decoder(func_decoder_input)
                #decoder_output, decoder_hidden = self.decoder(decoder_input)
            decoder_output = F.dropout(decoder_output, p=self.dropout, training=self.training)
            step_neigh_lst = [lst[step] for lst in neigh_lst]
            #step_prev_inds = [lst[step] for lst in prev_inds]

            #neigh_output = torch.zeros(batch_size, self.att_outer_size).float().cuda()
            step_neigh_mask = torch.cuda.ByteTensor(batch_size, src_len).fill_(1)
            for x in range(batch_size):
                #neigh_output[x] = encoder_outputs[x, step_neigh_lst[x], :].sum(0)
                step_neigh_mask[x, step_neigh_lst[x]] = 0
            step_neigh_bias = step_neigh_mask.float() * -1e9
            neigh_output = self.neighbor_select(encoder_outputs, decoder_output, step_neigh_bias)
            
            #step_prev = leaf_hidden[torch.arange(batch_size), step_prev_inds].unsqueeze(0)
            func_logits = self.trg_word_map(decoder_output)
            #func_decoder_output = torch.cat((decoder_output, encoder_outputs[torch.arange(batch_size), par_inds[:, step], :].unsqueeze(0)), dim=2)
            choose_prob = F.log_softmax(self.choose(decoder_output), dim=2).squeeze(0)
            choose_probs[:, step, :] = choose_prob
            decoder_output = torch.cat((decoder_output, neigh_output.unsqueeze(0)), dim=2)

            dot = self.attention_select(encoder_outputs, decoder_output.transpose(0, 1), step_neigh_bias)
            func_logits[:, :, 1] = func_logits[:, :, 1] + (func_mask[:, step] * -1e9).unsqueeze(0)
            func_logits[:, :, 0] = -1e9
            finished = [x for x in range(batch_size) if step_neigh_lst[x]]
            dot[finished] = F.log_softmax(dot[finished], dim=2)
            #dot[finished] = F.log_softmax(dot[finished], dim=2) + choose_prob[finished, 0].unsqueeze(1).unsqueeze(1)
            #func_logits = F.log_softmax(func_logits.squeeze(0), dim=1) + choose_prob[:, 1].unsqueeze(1)
            func_logits = F.log_softmax(func_logits.squeeze(0), dim=1)
            decoder_outputs[:, step, :] = torch.cat((func_logits, dot.squeeze(1)), dim=1)
            #decoder_outputs[:, step, :] = F.log_softmax(torch.cat((func_logits.squeeze(0), dot.squeeze(1)), dim=1), dim=1)
            func_ind = trg_seqs[:, step] < self.trg_word_vocab_size
            content_ind = trg_seqs[:, step] >= self.trg_word_vocab_size
            decoder_input = torch.zeros(batch_size, self.att_outer_size).cuda()
            #func_decoder_input = torch.zeros(batch_size, self.att_outer_size).cuda()
            decoder_input[func_ind] = self.trg_word_embed(trg_seqs[:, step][func_ind].cuda())
            #func_decoder_input[func_ind] = self.trg_word_embed(trg_seqs[:, step][func_ind].cuda())
            decoder_input[content_ind] = ans_embed[content_ind, trg_seqs[:, step][content_ind] - self.trg_word_vocab_size, :]
            #func_decoder_input[content_ind] = encoder_input[content_ind, trg_seqs[:, step][content_ind] - self.trg_word_vocab_size, :]
            #decoder_input = ans_embed[torch.arange(batch_size), trg_seqs[:, step], :].unsqueeze(0)
            decoder_input = decoder_input.unsqueeze(0)
            #func_decoder_input = func_decoder_input.unsqueeze(0)

        return decoder_outputs, choose_probs

    def attention_select(self, encoder_outputs, decoder_output, attention_bias):
        decoder_output = self.trg_key(decoder_output)
        #decoder_output = self.trg_key_fc2(F.relu(self.trg_key_fc1(decoder_output)))
        dot = torch.bmm(decoder_output, encoder_outputs.transpose(1, 2))
        dot = dot + attention_bias[:, None, :]
        return dot

    def neighbor_select(self, encoder_outputs, decoder_output, step_neigh_bias):
        '''
        neigh_select_dot = torch.bmm(self.neigh_map(decoder_output).transpose(0, 1), encoder_outputs.transpose(1, 2))
        neigh_select_dot = neigh_select_dot + step_neigh_bias[:, None, :]
        #neigh_select_weight = F.softmax(neigh_select_dot, dim=2)
        neigh_select_weight = F.sigmoid(neigh_select_dot)
        '''
        neigh_select_weight = (step_neigh_bias == 0).float().unsqueeze(1)
        neigh_output = torch.bmm(neigh_select_weight, encoder_outputs).squeeze(1)
        return neigh_output

    def generate(self, src_seqs, src_lengths, beam_size, top_k):
        batch_size = src_seqs.size(0)

        # embed tokens plus positions
        #encoder_input = self.word_embed(src_seqs[:, :, 0].cuda())
        encoder_input = torch.cat((self.word_embed(src_seqs[:, :, 0].cuda()),
                                   self.dep_embed(src_seqs[:, :, 2].cuda()),
                                   self.pos_embed(src_seqs[:, :, 1].cuda())),
                                  dim=2)
        attention_mask = torch.arange(src_seqs.size(1)).expand(batch_size, src_seqs.size(1)) >= torch.LongTensor(src_lengths).expand(src_seqs.size(1), batch_size).transpose(0, 1)
        attention_bias = attention_mask.float() * -1e9
        attention_bias = attention_bias.cuda()

        x = F.dropout(encoder_input, p=self.dropout, training=self.training)

        for self_attention, ffn, norm1, norm2 in zip(self.self_attention_blocks,
                                                     self.ffn_blocks,
                                                     self.norm1_blocks,
                                                     self.norm2_blocks):
            y = self_attention(norm1(x), None, attention_bias)
            x = residual(x, y, self.dropout, self.training)
            y = ffn(norm2(x))
            x = residual(x, y, self.dropout, self.training)
        encoder_outputs = self.out_norm(x)

        init_hidden = encoder_outputs.clone()
        init_hidden[attention_mask.unsqueeze(2).expand(batch_size, src_seqs.size(1), self.att_outer_size)] = 0
        init_hidden = init_hidden.sum(1).unsqueeze(0)
        decoder_hidden = self.init_hidden(init_hidden)

        decoder_input = torch.zeros_like(encoder_outputs[:, 0, :]).unsqueeze(1) 
        src_length = encoder_outputs.size(1)
        beam_size = min(beam_size, src_length)

        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        dot = self.attention_select(encoder_outputs, decoder_output, attention_bias)
        

        logits = F.log_softmax(dot.squeeze(1), dim=1)
        beam = torch.zeros(beam_size, src_length).long().cuda()
        beam_probs = torch.zeros(beam_size).float().cuda()
        logprobs, argtop = torch.topk(logits, beam_size, dim=1)
        argtop = argtop.squeeze(0)
        beam[:, 0] = argtop
        beam_probs = logprobs.clone().squeeze(0)
        
        decoder_hiddens = decoder_hidden[0].expand(1, beam_size, self.rnn_hidden_size).contiguous()
        decoder_cells = decoder_hidden[1].expand(1, beam_size, self.rnn_hidden_size).contiguous()
        decoder_input = encoder_outputs[0, argtop, :].unsqueeze(0)
        encoder_outputs = encoder_outputs.expand(beam_size, src_length, self.att_outer_size).contiguous()
        attention_bias = attention_bias.expand(beam_size, src_length).contiguous()
        
        mask = torch.cuda.ByteTensor(beam_size, src_length).fill_(0)
        mask[torch.arange(beam_size), beam[:, 0]] = 1
        for t in range(src_length - 1):
            top_k = min(top_k, src_length - t)

            decoder_output, decoder_hidden = self.decoder(decoder_input, (decoder_hiddens, decoder_cells))
            dot = self.attention_select(encoder_outputs, decoder_output.transpose(0, 1), attention_bias).squeeze(1)
            dot[mask] = -np.inf

            logits = F.log_softmax(dot, dim=1)
            logprobs, argtop = torch.topk(logits, top_k, dim=1)
            
            total_probs = beam_probs.unsqueeze(1).expand(beam_size, top_k).contiguous()
            total_probs = total_probs + logprobs
            best_probs, best_args = total_probs.view(-1).topk(beam_size)
            
            _decoder_hiddens = decoder_hiddens.clone()
            _decoder_cells = decoder_cells.clone()

            last = (best_args / top_k)
            curr = (best_args % top_k)
            beam = beam[last]
            beam_probs = best_probs
            beam[:, t+1] = argtop[last, curr]
            mask = mask[last]
            mask[torch.arange(beam_size), beam[:, t+1]] = 1
            
            decoder_hiddens = _decoder_hiddens[:, last, :]
            decoder_cells = _decoder_cells[:, last, :]
            decoder_input = encoder_outputs[0, beam[:, t+1], :].unsqueeze(0)
        
        best, best_arg = beam_probs.max(0)
        best_order = beam[best_arg]
        generations = src_seqs[0, best_order, 0]
        print(best.item() / src_length)
        return generations, best

    def tree_generate(self, src_seqs, src_lengths, beam_size, top_k, order, graph):
        batch_size = src_seqs.size(0)
        adj_lst, root, neigh_lst, prev_inds, depths, func_mask = graph
        src_len = src_lengths[0]

        # embed tokens plus positions
        #encoder_input = self.word_embed(src_seqs[:, :, 0].cuda())
        word_embed = self.word_embed_map(self.word_embed(src_seqs[:, :, 0].cuda()))
        depth_embed = torch.cuda.LongTensor(depths)
        depth_embed[depth_embed > self.max_dep] = self.max_dep
        depth_embed = self.lvl_embed(depth_embed)
        dep_embed = self.dep_embed(src_seqs[:, :, 2].cuda())
        pos_embed = self.pos_embed(src_seqs[:, :, 1].cuda())
        space_embed = self.space_embed(src_seqs[:, :, 4].cuda())
        
        '''
        encoder_input = torch.cat((word_embed,
                                   self.dep_embed(src_seqs[:, :, 2].cuda()),
                                   self.pos_embed(src_seqs[:, :, 1].cuda())),
                                  dim=2) + depth_embed
        '''
        encoder_input = word_embed + dep_embed + pos_embed + depth_embed + space_embed
        attention_mask = torch.arange(src_seqs.size(1)).expand(batch_size, src_seqs.size(1)) >= torch.LongTensor(src_lengths).expand(src_seqs.size(1), batch_size).transpose(0, 1)
        attention_bias = attention_mask.float() * -1e9
        attention_bias = attention_bias.cuda()

        neigh_mask = torch.ByteTensor(batch_size, src_len, src_len).fill_(0)
        for y in range(len(adj_lst)):
            neigh_mask[0, y, adj_lst[y]] = 1
            neigh_mask[0, adj_lst[y], y] = 1
        neigh_mask[:, torch.arange(src_len), torch.arange(src_len)] = 1 

        graph_bias = (~neigh_mask | attention_mask[:, None, :]).float() * -1e9
        graph_bias = graph_bias.cuda()

        #x = F.dropout(encoder_input, p=self.dropout, training=self.training)
        x = encoder_input
        layer_output = [self.out_norm(x)]

        for self_attention, ffn, norm1, norm2 in zip(self.self_attention_blocks,
                                                     self.ffn_blocks,
                                                     self.norm1_blocks,
                                                     self.norm2_blocks):
            #y = self_attention(norm1(x), None, attention_bias)
            y = self_attention(norm1(x), None, graph_bias, None, None)
            x = residual(x, y, self.dropout, self.training)
            y = ffn(norm2(x))
            x = residual(x, y, self.dropout, self.training)
            layer_output.append(self.out_norm(x))

        #encoder_outputs = self.out_norm(x)
        encoder_outputs = sum(layer_output)

        init_hidden = encoder_outputs.clone()
        init_hidden[attention_mask.unsqueeze(2).expand(batch_size, src_seqs.size(1), self.att_outer_size)] = 0
        init_hidden = init_hidden.sum(1).unsqueeze(0)
        decoder_hidden = self.init_hidden(init_hidden)

        decoder_input = torch.zeros_like(encoder_outputs[:, 0, :]).unsqueeze(1) 
        src_length = encoder_outputs.size(1)
 
        '''
        leaf_encoder_input = torch.zeros(1, 1, self.att_outer_size).cuda()
        leaf_output, leaf_hidden = self.leaf_encoder(leaf_encoder_input)
        '''

        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        step_neigh_mask = torch.cuda.ByteTensor(1, src_len).fill_(1)
        step_neigh_mask[0, neigh_lst[0]] = 0
        step_neigh_bias = step_neigh_mask.float() * -1e9
        neigh_output = self.neighbor_select(encoder_outputs, decoder_output, step_neigh_bias)

        func_logits = self.trg_word_map(decoder_output)
        choose_prob = F.log_softmax(self.choose(decoder_output), dim=2).squeeze(0)
        decoder_output = torch.cat((decoder_output, neigh_output.unsqueeze(0)), dim=2)
        dot = self.attention_select(encoder_outputs, decoder_output, step_neigh_bias).squeeze(1)
        #dot = F.log_softmax(torch.cat((func_logits.squeeze(0), dot.squeeze(1)), dim=1), dim=1)
        dot = F.log_softmax(dot, dim=1) + choose_prob[:, 0].unsqueeze(1)
        func_logits = F.log_softmax(func_logits.squeeze(0), dim=1) + choose_prob[:, 1].unsqueeze(1)
        dot = torch.cat((func_logits, dot.squeeze(1)), dim=1)
        par = root
        children = adj_lst[par] + [par]
        mask = [y + self.trg_word_vocab_size for y in range(src_length) if y not in children]
        dot[:, mask] = -np.inf
        dot[:, 1] = -np.inf
        dot[:, 0] = -np.inf

        non_leaf = max(1, sum(1 for x in adj_lst if len(x) > 0))
        top_k = min(top_k, src_length)
        if order == 'hybrid':
            decode_len = max(src_lengths[0] * 2, src_lengths[0] + 5)
            logits = F.log_softmax(dot, dim=1)
            logprobs, argtop = torch.topk(logits, top_k, dim=1)
            beam = torch.zeros(beam_size, decode_len).long().cuda()
            beam_probs = torch.zeros(beam_size).float().cuda()
            beam_probs[:] = -np.inf
            beam[:top_k, 0] = argtop.squeeze(0)
            beam_probs[:top_k] = logprobs[0]
        elif order == 'depth':
            logits = F.log_softmax(dot, dim=1)
            beam = torch.zeros(beam_size, src_length + non_leaf).long().cuda()
            beam_probs = torch.zeros(beam_size).float().cuda()
            beam_probs[:] = -np.inf
            beam[0, 0] = root
            beam_probs[0] = logits[0, root]
            decode_len = src_length + non_leaf - 1

        decoder_hiddens = decoder_hidden[0].expand(1, beam_size, self.rnn_hidden_size).contiguous()
        decoder_cells = decoder_hidden[1].expand(1, beam_size, self.rnn_hidden_size).contiguous()
        #leaf_decoder_hiddens = leaf_hidden[0].expand(1, beam_size, self.rnn_hidden_size).contiguous()
        #leaf_decoder_cells = leaf_hidden[1].expand(1, beam_size, self.rnn_hidden_size).contiguous()
        decoder_input = encoder_outputs[0, [root] * beam_size, :].unsqueeze(0)
        encoder_outputs = encoder_outputs.expand(beam_size, src_length, self.att_outer_size).contiguous()
        attention_bias = attention_bias.expand(beam_size, src_length).contiguous()
        
        current = np.array([Node(root) for x in range(beam_size)])
        if order == 'hybrid':
            for x in range(beam_size):
                node = Node(beam[x, 0].item())
                node.parent = current[x]

        for t in range(decode_len - 1):
            decoder_output, decoder_hidden = self.decoder(decoder_input, (decoder_hiddens, decoder_cells))

            neigh_output = torch.zeros(beam_size, self.att_outer_size).float().cuda()
            mask = torch.cuda.ByteTensor(beam_size, src_length + self.trg_word_vocab_size).fill_(0)
            step_neigh_mask = torch.cuda.ByteTensor(beam_size, src_length).fill_(1)
            unfinished = []
            for x in range(beam_size):
                if order == 'hybrid':
                    par = current[x].name % self.trg_word_vocab_size
                    prev = [ch.name for ch in current[x].children]
                    children = adj_lst[par] + [par]
                    curr_children = current[x].children
                    #if len(prev) == len(children):
                    if beam[x, t] == 1:
                        i = 0
                        while i < len(curr_children):
                            if prev[i] >= self.trg_word_vocab_size and \
                               prev[i] - self.trg_word_vocab_size != par and \
                               adj_lst[prev[i] - self.trg_word_vocab_size]:
                                current[x] = current[x].children[i]
                                break
                            else:
                                i += 1
                        if i == len(curr_children):
                            while current[x].parent is not None:
                                curr_name = current[x].name
                                current[x] = current[x].parent
                                names = [ch.name for ch in current[x].children]
                                ind = names.index(curr_name)
                                ind += 1
                                found = False
                                while ind < len(names):
                                    if names[ind] != current[x].name and \
                                       names[ind] >= self.trg_word_vocab_size and \
                                       adj_lst[names[ind] - self.trg_word_vocab_size]:
                                        found = True
                                        current[x] = current[x].children[ind]
                                        break
                                    ind += 1
                                if found:
                                    break
                        par = current[x].name % self.trg_word_vocab_size
                        prev = [ch.name for ch in current[x].children]
                        children = adj_lst[par] + [par]
                        _mask = [y + self.trg_word_vocab_size for y in range(src_length) if y not in children]
                    else:
                        _mask = [y + self.trg_word_vocab_size for y in range(src_length) if y not in children or y + self.trg_word_vocab_size in prev]
                        prev_cont = [ch.name for ch in current[x].children if ch.name >= self.trg_word_vocab_size]
                        if len(prev_cont) < len(children):
                            unfinished.append(x)
                    mask[x, _mask] = 1
                    neigh_select = [ch for ch in children if ch not in prev]
                    #neigh_output[x] = encoder_outputs[x, neigh_select, :].sum(0)
                    step_neigh_mask[x, neigh_select] = 0
         
                elif order == 'depth':
                    par = current[x].name
                    prev = [ch.name for ch in current[x].children]
                    children = adj_lst[par] + [par]
                    if len(children) == 1 or (current[x].parent is not None and par == current[x].parent.name):
                        while current[x].parent is not None:
                            current[x] = current[x].parent
                            par = current[x].name
                            children = adj_lst[par] + [par]
                            prev = [ch.name for ch in current[x].children]
                            if len(prev) < len(children):
                                break
                    _mask = [y for y in range(src_length) if y not in children or y in prev]
                    mask[x, _mask] = 1
                    neigh_select = [ch for ch in children if ch not in prev]
                    neigh_output[x] = encoder_outputs[x, neigh_select, :].sum(0)

            step_neigh_bias = step_neigh_mask.float() * -1e9
            neigh_output = self.neighbor_select(encoder_outputs, decoder_output, step_neigh_bias)

            func_logits = self.trg_word_map(decoder_output)
            choose_prob = F.log_softmax(self.choose(decoder_output), dim=2).squeeze(0)
            #dot = F.log_softmax(torch.cat((func_logits.squeeze(0), dot.squeeze(1)), dim=1), dim=1)
            decoder_output = torch.cat((decoder_output, neigh_output.unsqueeze(0)), dim=2)
            dot = self.attention_select(encoder_outputs, decoder_output.transpose(0, 1), step_neigh_bias).squeeze(1)
            dot = F.log_softmax(dot, dim=1) + choose_prob[:, 0].unsqueeze(1)
            func_logits = F.log_softmax(func_logits.squeeze(0), dim=1) + choose_prob[:, 1].unsqueeze(1)
            dot = torch.cat((func_logits, dot.squeeze(1)), dim=1)
            #dot = F.log_softmax(torch.cat((func_logits.squeeze(0), dot.squeeze(1)), dim=1), dim=1)
            dot[mask] = -np.inf
            dot[unfinished, 1] = -np.inf
            dot[:, 0] = -np.inf
            for x in range(beam_size):
                if x not in unfinished:
                    dot[x, self.trg_word_vocab_size:] = -np.inf
                else:
                    if choose_prob[x, 0] > choose_prob[x, 1]:
                        dot[x, :self.trg_word_vocab_size] = -np.inf
                    else:
                        dot[x, self.trg_word_vocab_size:] = -np.inf

            logits = F.log_softmax(dot, dim=1)
            logprobs, argtop = torch.topk(logits, top_k, dim=1)
            
            total_probs = beam_probs.unsqueeze(1).expand(beam_size, top_k).contiguous()
            total_probs = total_probs + logprobs
            best_probs, best_args = total_probs.view(-1).topk(beam_size)
            
            _decoder_hiddens = decoder_hiddens.clone()
            _decoder_cells = decoder_cells.clone()
            #_leaf_decoder_hiddens = leaf_decoder_hiddens.clone()
            #_leaf_decoder_cells = leaf_decoder_cells.clone()

            last = (best_args / top_k)
            curr = (best_args % top_k)
            beam = beam[last]
            beam_probs = best_probs
            beam[:, t+1] = argtop[last, curr]
            
            decoder_hiddens = _decoder_hiddens[:, last, :]
            decoder_cells = _decoder_cells[:, last, :]
            #leaf_decoder_hiddens = _leaf_decoder_hiddens[:, last, :]
            #leaf_decoder_cells = _leaf_decoder_cells[:, last, :]
            func_ind = beam[:, t+1] < self.trg_word_vocab_size
            content_ind = beam[:, t+1] >= self.trg_word_vocab_size
            decoder_input = torch.zeros(beam_size, self.att_outer_size).cuda()
            decoder_input[func_ind] = self.trg_word_embed(beam[:, t+1][func_ind].cuda())
            decoder_input[content_ind] = encoder_outputs[content_ind, beam[:, t+1][content_ind] - self.trg_word_vocab_size, :]
            #decoder_input = encoder_outputs[0, beam[:, t+1], :].unsqueeze(0)
            decoder_input = decoder_input.unsqueeze(0)

            new_curr = []
            for x in range(beam_size):
                curr = deepcopy(current[last[x].item()])
                node = Node(beam[x, t+1].item())
                node.parent = curr
                if order == 'hybrid':
                    new_curr.append(curr)
                elif order == 'depth':
                    new_curr.append(node)
            current = new_curr

        best, best_arg = beam_probs.max(0)
        best_order = beam[best_arg]
        best_tree = current[best_arg]
        while best_tree.parent is not None:
            best_tree = best_tree.parent

        def agg(tree):
            if not tree.children:
                return [tree.name]
            else:
                return list(itertools.chain.from_iterable([agg(ch) for ch in tree.children]))

        best_order = agg(best_tree)
        generations = [(src_seqs[0, x - self.trg_word_vocab_size, 0].item(), True) if x >= self.trg_word_vocab_size else (x, False) for x in best_order]
        print(best.item() / src_length)
        return generations, best, best_order


    def masked_loss(self, logits, target, mask):
        batch_size = logits.size(0)
        #log_probs_flat = F.log_softmax(logits, dim=2).view(-1, logits.size(-1))
        log_probs_flat = logits.view(-1, logits.size(-1))
        target_flat = target.contiguous().view(-1, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        losses = losses_flat.view(*target.size())
        losses = losses * mask.float()
        loss = losses.sum() / mask.float().sum()
        return loss, mask.float().sum()

    def loss(self, src_seqs, src_lengths, trg_seqs, trg_lengths, graph):
        batch_size = src_seqs.size(0)
        decoder_outputs, choose_probs = self.forward(src_seqs, src_lengths, trg_seqs, trg_lengths, graph)
        func_ind = trg_seqs < self.trg_word_vocab_size
        mask = torch.arange(trg_seqs.size(1)).expand(batch_size, trg_seqs.size(1)) < torch.LongTensor(trg_lengths).expand(trg_seqs.size(1), batch_size).transpose(0, 1)
        loss, count = self.masked_loss(decoder_outputs, trg_seqs.cuda(), mask.cuda())
        class_loss, _ = self.masked_loss(choose_probs, func_ind.long().cuda(), mask.cuda())
        return loss, class_loss
               

class T6_Transformer(nn.Module):
    def __init__(self, input_size, rnn_hidden_size, att_hidden_size, word_vectors, dictionary, dropout, num_layers, num_heads):
        super(T6_Transformer, self).__init__()
        self.rnn_hidden_size = rnn_hidden_size
        self.att_outer_size = 512
        self.att_hidden_size = att_hidden_size
        self.input_size = input_size
        self.word_vocab_size = len(dictionary['word']) + 1
        self.pos_vocab_size = len(dictionary['pos']) + 1
        self.dep_vocab_size = len(dictionary['dep']) + 1
        self.trg_word_vocab_size = len(dictionary['trg_word']) + 1
        self.word_embed = nn.Embedding(self.word_vocab_size, input_size)
        self.trg_word_embed = nn.Embedding(self.trg_word_vocab_size, input_size)
        self.word_embed_map = nn.Linear(input_size, self.att_outer_size)
        self.src_map = nn.Linear(rnn_hidden_size * 2, input_size)
        self.pos_embed = nn.Embedding(self.pos_vocab_size, input_size)
        self.dep_embed = nn.Embedding(self.dep_vocab_size, input_size)
        self.space_embed = nn.Embedding(5, self.att_outer_size)
        self.max_dep = 15
        self.lvl_embed = nn.Embedding(self.max_dep + 1, self.att_outer_size)
        self.word_embed.weight.data = torch.from_numpy(word_vectors.astype(np.float32))
        self.encoder = nn.LSTM(self.input_size, rnn_hidden_size, bidirectional=True)
        self.decoder = nn.LSTM(self.input_size, rnn_hidden_size)
        self.func_decoder = nn.LSTM(self.att_outer_size, rnn_hidden_size)
        self.trg_key = nn.Linear(rnn_hidden_size * 1 + self.att_outer_size, self.att_outer_size, bias=False)
        #self.trg_key_fc1 = nn.Linear(rnn_hidden_size * 1 + self.att_outer_size, att_hidden_size, bias=False)
        #self.trg_key_fc2 = nn.Linear(att_hidden_size, self.att_outer_size, bias=False)
        self.layer_agg = nn.Linear(self.att_outer_size * (num_layers + 1), self.att_outer_size)

        self.hidden_fc1 = nn.Linear(self.rnn_hidden_size * 2, att_hidden_size, bias=False)
        self.hidden_fc2 = nn.Linear(self.att_hidden_size, self.rnn_hidden_size, bias=False)
        self.cell_fc1 = nn.Linear(self.rnn_hidden_size * 2, att_hidden_size, bias=False)
        self.cell_fc2 = nn.Linear(self.att_hidden_size, self.rnn_hidden_size, bias=False)

        self.key_size = 50
        self.q_key = nn.Linear(rnn_hidden_size * 2, self.key_size)
        self.q_val = nn.Linear(rnn_hidden_size * 2, rnn_hidden_size)
        self.a_key = nn.Linear(rnn_hidden_size, self.key_size)
        self.p_key = nn.Linear(rnn_hidden_size, self.key_size)

        self.dictionary = dictionary
        self.dropout = dropout
        self.drop = nn.Dropout(p=dropout)

        self.out = nn.Linear(self.rnn_hidden_size * 1, self.trg_word_vocab_size)
        self.choose = nn.Linear(self.rnn_hidden_size * 1, 2)

        for names in self.decoder._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.decoder, name)
                n = bias.size(0)
                start, end = n//4, n//2 
                bias.data[start:end].fill_(0.)

    def post_parameters(self):
        return []

    def flatten_parameters(self):
        self.decoder.flatten_parameters()

    def hidden_transform(self, hidden):
        return self.hidden_fc2(F.relu(self.hidden_fc1(hidden)))

    def cell_transform(self, cell):
        return self.cell_fc2(F.relu(self.cell_fc1(cell)))

    def init_hidden(self, src_hidden):
        hidden = self.hidden_transform(src_hidden)
        cell = self.cell_transform(src_hidden)
        return (hidden, cell)

    def attention(self, decoder_hidden, src_hidden, src_lengths):
        a_key = self.a_key(decoder_hidden[0].squeeze(0))
        length = src_hidden.size(1)

        q_key = self.q_key(src_hidden)
        q_value = self.q_val(src_hidden)

        q_energy = torch.bmm(q_key, a_key.unsqueeze(2)).squeeze(2)
        q_mask = torch.arange(length).long().cuda().repeat(src_hidden.size(0), 1) < torch.cuda.LongTensor(src_lengths).repeat(length, 1).transpose(0, 1)
        q_energy[~q_mask] = -np.inf
        #q_weights = F.sigmoid(q_energy).unsqueeze(1)
        q_weights = F.softmax(q_energy, dim=1).unsqueeze(1)
        q_context = torch.bmm(q_weights, q_value)

        return q_context

    def forward(self, src_seqs, src_lengths, trg_seqs, trg_lengths, graph):
        batch_size = src_seqs.size(0)
        src_seqs = src_seqs.cuda()
        label_seqs = [g[0] for g in graph]
        src_inds = [g[1] for g in graph]

        max_trg_len = max(trg_lengths)
        src_len = src_seqs.size(1)
        for x in range(batch_size):
            label_seqs[x] = label_seqs[x] + [0] * (max_trg_len - len(label_seqs[x]))
            src_inds[x] = src_inds[x] + [0] * (max_trg_len - len(src_inds[x]))

        src_lengths, perm_idx = torch.LongTensor(src_lengths).sort(descending=True)
        src_lengths = src_lengths.tolist()
        src_seqs = src_seqs[perm_idx]
        trg_seqs = trg_seqs[perm_idx]
        label_seqs = torch.cuda.LongTensor(label_seqs)[perm_idx]
        src_inds = torch.cuda.LongTensor(src_inds)[perm_idx]

        src_word_embed = self.drop(self.word_embed(src_seqs[:, :, 0]))
        src_pos_embed = self.drop(self.pos_embed(src_seqs[:, :, 1]))
        src_dep_embed = self.drop(self.dep_embed(src_seqs[:, :, 2]))
        src_embed = src_word_embed + src_pos_embed + src_dep_embed
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)

        decoder_hidden = self.init_hidden(src_hidden[:, 0, :].unsqueeze(0))
        decoder_input = self.drop(self.trg_word_embed(torch.cuda.LongTensor([1] * batch_size))).unsqueeze(0)
        decoder_outputs = torch.zeros(batch_size, max_trg_len, self.trg_word_vocab_size).cuda()
        choose_probs = torch.zeros(batch_size, max_trg_len, 2).cuda()
        for step in range(max_trg_len):
            #context = self.attention(decoder_hidden, src_hidden, src_lengths)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            choose_probs[:, step, :] = self.choose(decoder_output.squeeze(1))
            #decoder_output = torch.cat((self.drop(decoder_output.transpose(0, 1)), self.drop(context)), dim=2)
            decoder_outputs[:, step, :] = self.out(decoder_output.squeeze(1))

            func_ind = label_seqs[:, step] == 1
            content_ind = label_seqs[:, step] == 0
            decoder_input = torch.zeros(batch_size, self.input_size).cuda()
            decoder_input[func_ind] = self.trg_word_embed(trg_seqs[:, step][func_ind].cuda())
            decoder_input[content_ind] = self.src_map(src_hidden[content_ind, src_inds[:, step][content_ind], :])
            decoder_input = decoder_input.unsqueeze(0)

        loss_trg = trg_seqs.clone()
        loss_trg[label_seqs != 1] = 0
        return decoder_outputs, choose_probs, label_seqs, loss_trg

    def neighbor_select(self, encoder_outputs, decoder_output, step_neigh_bias):
        '''
        neigh_select_dot = torch.bmm(self.neigh_map(decoder_output).transpose(0, 1), encoder_outputs.transpose(1, 2))
        neigh_select_dot = neigh_select_dot + step_neigh_bias[:, None, :]
        #neigh_select_weight = F.softmax(neigh_select_dot, dim=2)
        neigh_select_weight = F.sigmoid(neigh_select_dot)
        '''
        neigh_select_weight = (step_neigh_bias == 0).float().unsqueeze(1)
        neigh_output = torch.bmm(neigh_select_weight, encoder_outputs).squeeze(1)
        return neigh_output

    def generate(self, src_seqs, src_lengths, beam_size, top_k, order, graph):
        batch_size = src_seqs.size(0)
        src_seqs = src_seqs.cuda()
        label_seqs = [g[0] for g in graph]
        src_inds = [g[1] for g in graph]

        src_len = src_seqs.size(1)
        '''
        max_trg_len = max(trg_lengths)
        for x in range(batch_size):
            label_seqs[x] = label_seqs[x] + [0] * (max_trg_len - len(label_seqs[x]))
            src_inds[x] = src_inds[x] + [0] * (max_trg_len - len(src_inds[x]))
        '''
        src_word_embed = self.drop(self.word_embed(src_seqs[:, :, 0]))
        src_pos_embed = self.drop(self.pos_embed(src_seqs[:, :, 1]))
        src_dep_embed = self.drop(self.dep_embed(src_seqs[:, :, 2]))
        src_embed = src_word_embed + src_pos_embed + src_dep_embed
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_hidden[:, 0, :].unsqueeze(0))
        decoder_input = self.trg_word_embed(torch.cuda.LongTensor([1] * batch_size)).unsqueeze(0)
        
        src_ind = -1
        generations = []
        best = 0
        while True:
            #context = self.attention(decoder_hidden, src_hidden, src_lengths)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            choose_probs = F.log_softmax(self.choose(decoder_output.squeeze(1)), dim=1)
            #decoder_output = torch.cat((self.drop(decoder_output.transpose(0, 1)), self.drop(context)), dim=2)
            decoder_outputs = F.log_softmax(self.out(decoder_output.squeeze(1)), dim=1)    
            if choose_probs[0, 0] > choose_probs[0, 1] and src_ind < src_len - 1:
                src_ind += 1
                decoder_input = self.src_map(src_hidden[:, src_ind, :]).unsqueeze(1)
                generations.append((src_seqs[0, src_ind, 0].item(), True))
                best += choose_probs[0, 0]
            else:
                if src_ind < src_len - 1:
                    decoder_outputs[:, 1] = -np.inf
                best_prob, best_arg = decoder_outputs.max(1)
                decoder_input = self.trg_word_embed(best_arg).unsqueeze(1)
                generations.append((best_arg[0].item(), False))
                best += choose_probs[0, 1] + best_prob[0]
                if best_arg[0].item() == 1:
                    break

        return generations, best
        '''
        logits = F.log_softmax(dot.squeeze(1), dim=1)
        beam = torch.zeros(beam_size, src_length).long().cuda()
        beam_probs = torch.zeros(beam_size).float().cuda()
        logprobs, argtop = torch.topk(logits, beam_size, dim=1)
        argtop = argtop.squeeze(0)
        beam[:, 0] = argtop
        beam_probs = logprobs.clone().squeeze(0)
        
        decoder_hiddens = decoder_hidden[0].expand(1, beam_size, self.rnn_hidden_size).contiguous()
        decoder_cells = decoder_hidden[1].expand(1, beam_size, self.rnn_hidden_size).contiguous()
        decoder_input = encoder_outputs[0, argtop, :].unsqueeze(0)
        encoder_outputs = encoder_outputs.expand(beam_size, src_length, self.att_outer_size).contiguous()
        attention_bias = attention_bias.expand(beam_size, src_length).contiguous()
        
        mask = torch.cuda.ByteTensor(beam_size, src_length).fill_(0)
        mask[torch.arange(beam_size), beam[:, 0]] = 1
        for t in range(src_length - 1):
            top_k = min(top_k, src_length - t)

            decoder_output, decoder_hidden = self.decoder(decoder_input, (decoder_hiddens, decoder_cells))
            dot = self.attention_select(encoder_outputs, decoder_output.transpose(0, 1), attention_bias).squeeze(1)
            dot[mask] = -np.inf

            logits = F.log_softmax(dot, dim=1)
            logprobs, argtop = torch.topk(logits, top_k, dim=1)
            
            total_probs = beam_probs.unsqueeze(1).expand(beam_size, top_k).contiguous()
            total_probs = total_probs + logprobs
            best_probs, best_args = total_probs.view(-1).topk(beam_size)
            
            _decoder_hiddens = decoder_hiddens.clone()
            _decoder_cells = decoder_cells.clone()

            last = (best_args / top_k)
            curr = (best_args % top_k)
            beam = beam[last]
            beam_probs = best_probs
            beam[:, t+1] = argtop[last, curr]
            mask = mask[last]
            mask[torch.arange(beam_size), beam[:, t+1]] = 1
            
            decoder_hiddens = _decoder_hiddens[:, last, :]
            decoder_cells = _decoder_cells[:, last, :]
            decoder_input = encoder_outputs[0, beam[:, t+1], :].unsqueeze(0)
        
        best, best_arg = beam_probs.max(0)
        best_order = beam[best_arg]
        generations = src_seqs[0, best_order, 0]
        print(best.item() / src_length)
        return generations, best
        '''

    def tree_generate(self, src_seqs, src_lengths, beam_size, top_k, order, graph):
        return self.generate(src_seqs, src_lengths, beam_size, top_k, order, graph)
        
    def masked_loss(self, logits, target, mask):
        batch_size = logits.size(0)
        log_probs_flat = F.log_softmax(logits, dim=2).view(-1, logits.size(-1))
        #log_probs_flat = logits.view(-1, logits.size(-1))
        target_flat = target.contiguous().view(-1, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        losses = losses_flat.view(*target.size())
        losses = losses * mask.float()
        loss = losses.sum() / mask.float().sum()
        return loss, mask.float().sum()

    def loss(self, src_seqs, src_lengths, trg_seqs, trg_lengths, graph):
        batch_size = src_seqs.size(0)
        decoder_outputs, choose_probs, label_seqs, loss_trg = self.forward(src_seqs, src_lengths, trg_seqs, trg_lengths, graph)
        mask = torch.arange(trg_seqs.size(1)).expand(batch_size, trg_seqs.size(1)) < torch.LongTensor(trg_lengths).expand(trg_seqs.size(1), batch_size).transpose(0, 1)
        func_mask = label_seqs == 1
        loss, count = self.masked_loss(decoder_outputs, loss_trg.cuda(), mask.cuda() & func_mask)
        class_loss, _ = self.masked_loss(choose_probs, label_seqs.long().cuda(), mask.cuda())
        return loss, class_loss
       
