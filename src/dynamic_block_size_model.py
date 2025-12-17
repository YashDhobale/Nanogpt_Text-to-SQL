# Base GPT model which uses dynamic block sizes based on semicolon positions
import os
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import uvicorn

CORS_ORIGINS = os.environ.get("NANOGPT_CORS_ORIGINS", "*").split(",")

app = FastAPI(title="NanoGPT Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 50

class GenerateResponse(BaseModel):
    generated_text: str


# hyperparameters
batch_size = 1 # how many independent sequences will we process in parallel?
block_size = 2048 # what is the maximum context length for predictions?
current_iteration = 0 # Renamed 'iter' to 'current_iteration'
eval_interval = 400*4
learning_rate = 3*(1e-5)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 128
n_head = 8
n_layer = 8
dropout = 0.0
# ------------

# Load input file from project public/ folder using project-root resolution
project_root = Path(__file__).resolve().parent.parent
input_path = project_root / 'public' / 'synthetic_text_to_sql_with_schema_modified.txt'

if not input_path.exists():
    raise FileNotFoundError(f"inputsql.txt not found at {input_path}")

with open(input_path, 'r', encoding='utf-8') as f:
    text = f.read()

# print(__builtins__.len(text))
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = __builtins__.len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*__builtins__.len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split,checkpoints,current_iteration):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    #print(checkpoints[current_iteration])
    start = checkpoints[current_iteration]
    end = checkpoints[current_iteration+1]
    x = torch.stack([data[start:end]])
    y = torch.stack([data[start+1:end+1]])

    # Move to device and return
    x = x.to(device)
    y = y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(checkpoints):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split,checkpoints,current_iteration+k)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        emb_device = self.token_embedding_table.weight.device
        idx = idx.to(emb_device)
        # idx is (B, T) array of indices in the current context
        for count in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            # idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) Multinomial Sampling Approach
            idx_next = torch.argmax(logits, dim=-1, keepdim=True) # (B, 1) Greedy Approach
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            # terminating condition for ";" which is 15
            #if idx_next[0][0]==torch.tensor(15):
            #  break

        return idx

checkpoints = []
# Load input file from project public/ folder using project-root resolution
project_root = Path(__file__).resolve().parent.parent
input_path = project_root / 'public' / 'semicolon_indices.txt'

if not input_path.exists():
    raise FileNotFoundError(f"semicolon_indices.txt not found at {input_path}")

with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        checkpoints.append(int(line.strip()))

model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

while (current_iteration < len(checkpoints)-10): # Use renamed variable

  # every once in a while evaluate the loss on train and val sets
    if current_iteration % eval_interval == 0 or current_iteration==block_size*16: # Use renamed variable
        losses = estimate_loss(checkpoints)
        print(f"step {current_iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}") # Use renamed variable
        text2="How many accidents have been recorded for SpaceX and Blue Origin rocket launches?# CREATE TABLE Accidents (id INT, launch_provider VARCHAR(255), year INT, description TEXT); INSERT INTO Accidents (id, launch_provider, year, description) VALUES (1, 'SpaceX', 2015, 'Falcon 9 explosion'), (2, 'Blue Origin', 2011, 'Propulsion system failure'), (3, 'SpaceX', 2016, 'Falcon 9 explosion');$"
        context = torch.tensor(encode(text2), dtype=torch.long, device=device)
        context = torch.stack([context[:]])
        print(decode(m.generate(context, max_new_tokens=1024)[0].tolist()))

    #checkpoints.append(i)
    # print(device + str(current_iteration))
    xb, yb=get_batch('train',checkpoints=checkpoints,current_iteration=current_iteration)
    current_iteration=current_iteration+1 # Use renamed variable
    #print(checkpoints)
    #print(decode(xb[0].tolist()))
    #print(decode(yb[0].tolist()))

  # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Training complete.")

# FastAPI routes (defined at module level so they register when imported by uvicorn)
@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Main generate endpoint - called by frontend."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
    try:
        # create the tensor directly on the same device as the model to avoid device-mismatch
        context = torch.tensor(encode(request.prompt), dtype=torch.long, device=device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to encode prompt: {e}")
    # add batch dimension: shape becomes (1, prompt_length)
    context = context.unsqueeze(0)
    try:
        generated_ids = model.generate(context.to(device), max_new_tokens=max(request.max_new_tokens, 2000))[0].tolist()
        generated_text = decode(generated_ids)
    except Exception as e:
        # return a 500 with the exception message for easier debugging
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    return GenerateResponse(generated_text=generated_text)


# generate from the model
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
text2="How many accidents have been recorded for SpaceX and Blue Origin rocket launches?# CREATE TABLE Accidents (id INT, launch_provider VARCHAR(255), year INT, description TEXT); INSERT INTO Accidents (id, launch_provider, year, description) VALUES (1, 'SpaceX', 2015, 'Falcon 9 explosion'), (2, 'Blue Origin', 2011, 'Propulsion system failure'), (3, 'SpaceX', 2016, 'Falcon 9 explosion');$"
context = torch.tensor(encode(text2), dtype=torch.long, device=device)
context = torch.stack([context[:]])
print(decode(m.generate(context, max_new_tokens=1024)[0].tolist()))
