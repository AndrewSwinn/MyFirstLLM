import os.path
import configparser
import torch
import torch.nn as nn
from torch.nn import functional as F

config = configparser.ConfigParser()
config.read(os.path.join(os.getcwd(), 'config.ini'))
data_dir = config.get('directories', 'shakespeare')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data files
data_files = {'Train': 'train.csv', 'Test': 'test.csv', 'Val': 'validation.csv'}
texts = {key: open(os.path.join(data_dir,  file)).read() for key, file in data_files.items()}

# Build character encoder / decoders.
# (there is a tiktoken package available for most sophisticated tokens

chars = sorted(list(set(texts['Train'])))
vocab_size = len(chars)
stoi = {ch:i for i, ch in enumerate(chars)} # string to integer encoding scheme
itoc = {i:ch for i, ch in enumerate(chars)} # string to integer decoding scheme
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itoc[i] for i in l])

batch_size    = 64
block_size    = 256
max_iters     = 10000
eval_interval = 500
learning_rate = 3e-4
nembed        = 384
n_layer       = 6
n_head        = 6
dropout       = 0.2


tokens = {key: encode(text) for key, text in texts.items()}
data   = {key: torch.tensor(token_list, dtype=torch.long) for key, token_list in tokens.items()}


def get_batch(data):
    ix = torch.randint(high=(len(data) - block_size), size=(batch_size,))
    x =   torch.stack([data[i  :   i + block_size    ] for i in ix])
    y =   torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad
def estimate_loss():
    eval_iters = 10
    out = {}
    model.eval()
    for split in ['Train', 'Val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, y = get_batch(data[split])
            logit, loss = model(X,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key   = nn.Linear(nembed, head_size, bias=False)
        self.query = nn.Linear(nembed, head_size, bias=False)
        self.value = nn.Linear(nembed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size )))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads      = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(nembed, nembed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return(out)


class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        head_size = embed_size // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ff = FeedForward(embed_size)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x



class BigramLanguageModel(nn.Module):

    def __init__(self):  #
        super().__init__()
        # each token directy reads the logits for the next token from a lookup table
        self.token_embedding_table    = nn.Embedding(vocab_size, nembed)
        self.position_embedding_table = nn.Embedding(block_size, nembed)
        self.blocks = nn.Sequential(*[Block(embed_size=nembed, num_heads=n_head) for _ in range(n_layer)])
        self.lm_head                  = nn.Linear(nembed, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are tensors with dimensions (batch_size, block_size)
        B, T = idx.shape
        # logits is tensor with dimensions (batch_size, block_size, vocab_size)
        token_embeddings   = self.token_embedding_table(idx)
        positon_embeddings = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embeddings + positon_embeddings
        x = self.blocks(x)
        logits             = self.lm_head(x)   # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape
            logits = logits.view(b * t, c)
            targets = targets.view(-1)
            logits = logits.view(b * t, c)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # idx is tensor with dimensions (batch_size, block_size)
            logits, loss = self(idx_cond)
            # focus on the last character
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # samples from the probability distribution.
            idx_next = torch.multinomial(input=probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == '__main__':

    model = BigramLanguageModel()
    m = model.to(device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

    print('Before Training\n', decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=100)[0].tolist()))
    print()

    for iter in range(max_iters):

        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"Step {iter}: train loss {losses['Train']:.4f}  val loss {losses['Val']:.4f}")

        xb, yb = get_batch(data['Train'])
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), os.path.join('out', 'bigram_model.pt'))

    print()

    print('After Training\n', decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=400)[0].tolist()))
