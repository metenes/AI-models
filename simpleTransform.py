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

        # Input diveded into 3 latent parts 
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False )
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # fully connected out 
        self.fc_out = nn.Linear(heads * self.head_dim, emb_size, bias=False)

    def forward(self, values, keys, query, mask): 
        N = query.shape[0] # Number of trainig example = How many examples we will send in the same time = number of input vectors
        # the lenghts of the values will depend on the currently processing embedding 
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1] 

        # split embeddigns into self.heads pieces which contains self.head_dim
        values = values.reshape(N, value_len, self.heads, self.head_dim )
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        # Now we will dot product the quaries with keys which is called energy
        # einsum is just a easy way to do dot product
        energy = torch.einsum("nqhd,nkhd-->nhqk" , [query, keys])
        # energy = [N, heads, query_len, key_len] means, it shows how much attention must be given to each quary by each key
        # means defining the weights for current iteration 
        
        # mask 
        if mask is not None: 
            # means, we will use a mask which has upper triange matrix values set to - inf. (very small value)
            # mask == 0 means if masks value equals to zero, we will replace energy matrixs upper triangel by the - inf. 
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # self or dot product attention 
        attention = torch.softmax(energy / (self.emb_size ** (1/2)), dim= 3)
        # dim = 3 means normalizing with respec to key length 
        # energy = attention : (N, heads, query_len, key_len) , key_len is at index 3 

        out = torch.einsum("nhql,nlhd-->nqhd")
        # out = (N, quary_len, heads, head_dim) = quaries in different values 
        # because of the key_len == value_len, we will call it l 
        # values : [ N, value_len, heads, heads_dim ]
        # attention : [ N, heads, query_len, key_len ]
        # we will dot prod values with attention 

        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    # Transformer block will be consist of Multihead attention and Feed Forward
    def __init__( self, embed_size, heads, dropout, forward_expension ): 
        super(TransformerBlock, self).__init__()
        # Multihead attention just consist of dot prod attention 
        self.attention = selfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        # feed forward 
        # additionel neural network whichs weights have come from pre-training 
        self.feed_forward = nn.Sequential(
                nn.Linear(embed_size, forward_expension*embed_size), 
                nn.ReLU(), 
                nn.Linear(forward_expension*embed_size , embed_size) 
        )
        # drop out = zeros all the channels to prevent adaptation 
        self.dropout = nn.Dropout(dropout)

    def forward( self, value, key, query, mask ): 
        # first give q,k,v to self-attention 
        attention = self.attention(value, key, query, mask)
        # add skip connection and normalize 
        mid = self.dropout( self.norm1(query + attention) ) 
        # feed forward 
        forward = self.feed_forward(mid)
        out = self.dropout( self.norm2(mid+forward) )
        return out

class Encoder(nn.Module): 
    def __init__(self, 
                 src_vocab_size, 
                 embed_size, 
                 num_layers, 
                 heads, 
                 device, 
                 forward_expension, 
                 dropout, 
                 max_length # max length is the limit for vector number
                            ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        # nn.Embedding = A simple lookup table that stores embeddings of a fixed dictionary and size.
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [   TransformerBlock(embed_size, heads, dropout= dropout, forward_expension= forward_expension)  ]
        )
        # dropout same
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask): 
        # In x, we got N sentence and they has seq_lenght word vector
        N, seq_length = x.shape
        position = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
