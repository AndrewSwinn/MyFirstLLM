import os.path
import socket
import torch
import torch.nn as nn
from torch.nn import functional as F

if socket.gethostname() == 'LTSSL-sKTPpP5Xl':
    data_dir = 'C:\\Users\\ams90\\PycharmProjects\\ConceptsBirds\\data'
elif socket.gethostname() == 'LAPTOP-NA88OLS1':
    data_dir = 'D:\\data\\'
else:
    data_dir = '/home/bwc/ams90/datasets/caltecBirds/CUB_200_2011'

device = 'cpu' if torch.cuda.is_available() else 'cpu'

# Load data files
data_files = {'Train': 'train.csv', 'Test': 'test.csv', 'Val': 'validation.csv'}
texts = {key: open(os.path.join(data_dir, 'Text', file)).read() for key, file in data_files.items()}

# Build character encoder / decoders.
# (there is a tiktoken package available for most sophisticated tokens

chars = sorted(list(set(texts['Train'])))
vocab_size = len(chars)
stoi = {ch:i for i, ch in enumerate(chars)} # string to integer encoding scheme
itoc = {i:ch for i, ch in enumerate(chars)} # string to integer decoding scheme
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itoc[i] for i in l])

max_iters     = 10000
eval_interval = 500
nembed        = 32


tokens = {key: encode(text) for key, text in texts.items()}
data   = {key: torch.tensor(token_list, dtype=torch.long) for key, token_list in tokens.items()}

batch_size, block_size = 4,8
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





class BigramLanguageModel(nn.Module):

    def __init__(self):  #
        super().__init__()
        # each token directy reads the logits for the next token from a lookup table
        self.token_embedding_table    = nn.Embedding(vocab_size, nembed)
        self.position_embedding_table = nn.Embedding(block_size, nembed)
        self.lm_head                  = nn.Linear(nembed, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are tensors with dimensions (batch_size, block_size)
        B, T = idx.shape
        # logits is tensor with dimensions (batch_size, block_size, vocab_size)
        token_embeddings   = self.token_embedding_table(idx)
        positon_embeddings = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embeddings + positon_embeddings
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
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
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
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

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

    print()

    print('After Training\n', decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=200)[0].tolist()))
