import torch
import torch.nn as nn

# Self Attention 
class selfAttention(nn.Module): 
    def __init__(self, emb_size, heads): 
        """
        we have emb_size from the start of the process and we will split it
        heads is the number of parts we split emb size 
        """
        super(selfAttention, self).__init__()
        # Init
        self.emb_size = emb_size
        self.heads = heads
        self.head_dim = emb_size // heads

        # emb size must be divisible by heads
        assert (self.head_dim * heads == emb_size)

        # Input diveded into 3 parts 
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False )
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # fully connected out 
        self.fc_out = nn.Linear(heads * self.head_dim, emb_size, bias=False)

    def forward(self, values, keys, query, mask): 
        N = query.shape[0] # Number of trainig example = How many examples we will send in the same time 
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1] 
