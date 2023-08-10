#%% 
# Split / encode the data
import math
from typing import Optional, List
import torch
from torch import nn

class Prep_Quaries_Keys_Values(nn.Module): 
    def __init__(self, model_dim, heads, heads_dim, bias): 
        super().__init__()
        self.linear = nn.Linear(model_dim, heads * heads_dim, bias= bias)
        self.heads = heads
        self.heads_dim = heads_dim
    
    def forward(self, x): 
        # x : [seq_len, batch_size, model_dim], we save the first two dimensions to get adjust in view 
        head_shape = x.shape[:-1] 
        # We apply the linear transformation to the last dimension and split that into the heads. 
        x = self.linear(x)
        # view : Returns a new tensor with the same data as the self tensor but of a different shape.
        x = x.view(*head_shape, self.heads, self.heads_dim)
        # new x : [ seq_len, batch_size, self.heads, self.heads_dim ]
        return x 
#%% 
# Multihead Attention 

class MultiheadAttention(nn.Module): 
    def __init__(self, heads, model_dim, dropout = 0.1, bias = True):
        super().__init__()
        # Init
        self.head_dim = model_dim // heads # Number of features per head = head_dim
        self.heads = heads
        assert self.heads * self.head_dim == model_dim
        # Split the data 
        self.quary_fn = Prep_Quaries_Keys_Values(model_dim, heads, self.head_dim, bias)
        self.key_fn = Prep_Quaries_Keys_Values(model_dim, heads, self.head_dim, bias)
        self.value_fn = Prep_Quaries_Keys_Values(model_dim, heads, self.head_dim, bias=True) # bais must be added in values
        # Softmax func
        self.softmax = nn.Softmax(dim=1)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Out conv 
        self.out_conv = nn.Linear(model_dim, model_dim)
        # Scale factor 
        self.scale = 1 / math.sqrt(self.head_dim)

        self.attn = None 

    def get_scores(self, query, key): 
        # Score or Energy in this context is the dot product of Q and K 
        # author take seq_len to front, i and j indicate loop indices in dot prod
        return torch.einsum("ibhd,jbhd->ijbh", query, key)

    def prepare_mask(self, mask, query_shape, key_shape): 
        # mask : [seq_len_q, seq_len_k, batch_size] 
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        mask = mask.unsqueeze(-1)  
        # new mask : [seq_len_q, seq_len_k, batch_size, heads]
        return mask

    def forward(self, *, query, key, value, mask):
        # at inital:  query, key and value have shape [seq_len, batch_size, model_dim]
        seq_len, batch_size, _ = query.shape
        if mask is not None: 
            mask = self.prepare_mask(mask, query.shape, key.shape)

        # Split the heads and head_dim 
        query = self.quary_fn(query)
        key = self.key_fn(key)
        value = self.value_fn(value)
        # new query, key and value : [seq_len, batch_size, heads, head_dim]

        # score and attention 
        score = self.get_scores(query, key) * self.scale  # score calc.
        attn = self.softmax(score) # probability calc. 
        attn = self.dropout(attn) # droupout to not overfitting

        # half diagoenl of the mask will be changed to - inf. = sigmoid 0
        if mask is not None:  
            score = score.masked_fill(mask == 0, float('-inf')) 

        # values and attention 
        # Basically dot product of attention and values 
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)
        
        self.attn = attn.detach() # save the attention 

        # After calc. all the attn with different K,V,Qs , concat them to model_dim 
        x = x.reshape(seq_len, batch_size, -1) #  [seq_len, batch_size, model_dim = head*head_dim ]
        
        return self.out_conv(x)
