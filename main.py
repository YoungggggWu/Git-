import torch
import torch.optim as optim
from d2l import torch as d2l
from layers import *
from model import * 
from utils import *
from parse import *

import warnings
warnings.filterwarnings('ignore')
from time import time
# decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 128, 8, 0.5, 0)     
# decoder_blk.eval()
# state = [encoder_blk(X, valid_lens), valid_lens, [None]]
# print(decoder_blk(X, state)[0].shape)
args = parse_args()

# X = torch.ones(((2, 100, 24)))
# valid_lens = torch.tensor([10,12])
# encoder_block = EncoderBlock(args.key_size, args.query_size, args.value_size, args.num_hiddens, args.norm_shape,
#                                args.ffn_num_input, args.ffn_num_hiddens, args.num_heads, args.dropout,)
# encoder_block.eval()
# output = encoder_block(X, valid_lens)
# decoder_blk = DecoderBlock(args.key_size, args.query_size, args.value_size, args.num_hiddens, args.norm_shape,
#                                args.ffn_num_input, args.ffn_num_hiddens, args.num_heads, args.dropout, 0) 
# decoder_blk.eval()
# state = [encoder_block(X, valid_lens), valid_lens, [None]] 
# print(decoder_blk(X, state)+)
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4 
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]    

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout
)
decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout
)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
plt.show()

    