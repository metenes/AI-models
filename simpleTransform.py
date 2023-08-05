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
        # the lenghts of the values will depend on the currently processing embedding 
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1] 

        # split embeddigns into self.heads pieces which contains self.head_dim
        values = values.reshape(N, value_len, self.heads, self.head_dim )
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        # Now we will dot product the quaries with keys which is called energy
        # einsum is just a easy way to do dot product
        energy = torch.einsum("nqhd,nkhd-->nhqk" , [query, keys])
        
        # mask 
        if mask is not None: 
            # means, we will use a mask which has upper triange matrix values set to - inf. (very small value)
            # mask == 0 means if masks value equals to zero, we will replace energy matrixs upper triangel by the - inf. 
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # self or dot product attention 
        attention = torch.softmax(energy / (self.emb_size ** (1/2)), dim= 3)
        # dim = 3 means normalizing with respec to key length 
        # energy: (N, heads, query_len, key_len) , key_len is at index 3 
        # source sentence = key and target sentence = quary 
