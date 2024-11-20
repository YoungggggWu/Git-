import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run Transformer')
    parser.add_argument('--vacab_size', type=int, default=0, 
                        help='Number of vacab')
    parser.add_argument('--key_size', type=int, default=64,
                        help='Dim of key')
    parser.add_argument('--query_size', type=int, default=64,
                        help='Dim of query')
    parser.add_argument ('--value_size', type=int, default=64,
                         help= 'Dim of value')
    parser.add_argument('--num_hiddens', type=int, default=64,
                        help='Dim of Embeddings')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of Multi-Head Attention')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Message dropout')
    parser.add_argument('--norm_shape', nargs='+',type=int, default=[100, 64], 
                        help= 'Layer Normalized Shape')
    parser.add_argument('--ffn_num_input', type=int, default=64,
                        help='Dim of FFN Input')
    parser.add_argument('--ffn_num_hiddens', type=int, default=64,
                        help='Dim of FFN outputs')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of block')
    return parser.parse_args()
    
    
    
