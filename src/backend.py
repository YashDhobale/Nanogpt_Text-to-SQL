import os
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torch.nn import functional as F
import uvicorn
from transformers import get_cosine_schedule_with_warmup
import json
from datetime import datetime

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


class TrainRequest(BaseModel):
    text: str
    max_iters: Optional[int] = 500


class TrainResponse(BaseModel):
    status: str
    message: str


# hyperparameters
batch_size = 8 #how many independent sequences will we process in parallel?
block_size = 1024 # what is the maximum context length for predictions?
max_iters = 0
eval_interval = 400
learning_rate = 3*(1e-3)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
eval_iters = 100
n_embd = 128
n_head = 8
n_layer = 8
dropout = 0.0
# ------------

torch.manual_seed(1337)

# Load input file from project public/ folder using project-root resolution
project_root = Path(__file__).resolve().parent.parent
input_path = project_root / 'public' / 'synthetic_text_to_sql_with_schema.txt'

if not input_path.exists():
    raise FileNotFoundError(f"inputsql.txt not found at {input_path}")

with open(input_path, 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s: str):
    # safe encode: map unknown characters to 0 (fallback) to avoid KeyError
    # This keeps inference robust if the prompt contains characters not seen in training text.
    return [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):  #counts average loss over eval_iters batches
            X, Y = get_batch(split)
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
        # ensure input indices are on the same device as the embedding weights
        emb_device = self.token_embedding_table.weight.device
        idx = idx.to(emb_device)
        if targets is not None:
            targets = targets.to(emb_device)

        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        # create position indices on the same device to avoid device mismatch
        pos_indices = torch.arange(T, dtype=torch.long, device=emb_device)
        pos_emb = self.position_embedding_table(pos_indices) # (T,C)
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
        # idx is (B, T) array of indices in the current context
        # ensure idx is on the same device as the embedding weights
        emb_device = self.token_embedding_table.weight.device
        idx = idx.to(emb_device)
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
            # ensure idx_next is on the same device as idx before concatenating
            idx_next = idx_next.to(idx.device)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            # terminating condition for ";" which is 15
            # compare using Python int to avoid device mismatch between CPU/GPU tensors
            if idx_next[0, 0].item() == stoi[';']:
                break

        return idx

# create the model and move it to the selected device
model = BigramLanguageModel().to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer (kept here but training loop will only run when executed as a script)
weight_decay = 0.1     # strength of weight decay
warmup_steps = int(0.4*max_iters)    # you can also use int(0.03 * max_iters)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Scheduler: cosine decay
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_iters,)


# Training loop - only run when this file is executed via backrun.py
print("Starting training loop...")
for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters-1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        print(device + str(iter))
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

print("Training complete.")

# Utility: save model parameters to a text file (JSON readable)
def save_model_params(path: str):
    """Save the model's state_dict to a text file at `path`.
    Each parameter is written as a JSON array preceded by its name and shape.
    """
    try:
        sd = model.state_dict()
        with open(path, 'w', encoding='utf-8') as f:
            for k, v in sd.items():
                arr = v.detach().cpu().numpy()
                f.write(f"=== {k} {list(arr.shape)} ===\n")
                json.dump(arr.tolist(), f)
                f.write("\n\n")
        print(f"Model parameters saved to {path}")
    except Exception as e:
        print(f"Failed to save model parameters: {e}")

# attempt to save parameters from the top-level script training run
try:
    #ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    save_path = project_root / f"script_trained_params.txt"
    save_model_params(str(save_path))
except Exception as e:
    print(f"Warning: failed to save parameters after script training: {e}")

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


@app.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """Train the model on provided text."""
    global model, train_data, val_data, stoi, itos, device
    try:
        # Process the provided text to build vocabulary and encode data
        text = request.text
        
        # Encode the text
        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]
                
        # Training loop
        max_iters_to_run = request.max_iters if request.max_iters else 500
        for iter_num in range(max_iters_to_run):
            if iter_num % eval_interval == 0 or iter_num == max_iters_to_run-1:
                #losses = estimate_loss_train(get_batch_train)
                losses = estimate_loss()
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            xb, yb = get_batch('train')
            print(device + str(iter_num))
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # save parameters after training completes (timestamped)
        try:
            #ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            save_path = project_root / f"script_trained_params.txt"
            save_model_params(str(save_path))
        except Exception as e:
            print(f"Warning: failed to save parameters after /train: {e}")

        return TrainResponse(status="success", message=f"Model trained for {max_iters_to_run} iterations on {len(text)} characters")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# Example: generate from the model
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# text2 = "At which average price do I have NHPC Bond in my portfolio (investor_code=YA01)# CREATE TABLE investor_transactions (sr_no INT PRIMARY KEY, investor_name VARCHAR(50), investor_code VARCHAR(20), demat_code VARCHAR(20), instrument_type VARCHAR(10), instrument_name VARCHAR(100), isin VARCHAR(20), operation_type CHAR(1), quantity INT, amount DECIMAL(10,2), txn_date DATE);$"
# context = torch.tensor(encode(text2), dtype=torch.long, device=device)
# context = torch.stack([context])
# print(decode(model.generate(context, max_new_tokens=1024)[0].tolist()))





def load_model_params(path: str):
    """Load parameters from a file of a trained model and copy them
    into `model`'s state dict. The function does not get called automatically.

    Expected file format (as produced by save_model_params):
      === <param_name> [shape_list] ===\n
      <json array of values>\n\n
    This loader will parse each section, convert the JSON to a torch tensor,
    reshape it to the target parameter shape and copy it into the model.
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Parameter file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        sd = model.state_dict()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('==='):
                # parse the header: === key [shape] ===
                # extract the key between the === markers
                try:
                    header = line.strip('= ').strip()
                    # header starts with the key, shape follows; split once
                    parts = header.split(' ', 1)
                    key = parts[0]
                except Exception:
                    i += 1
                    continue

                # read the next non-empty line(s) as the JSON payload
                i += 1
                # skip blank lines
                while i < len(lines) and lines[i].strip() == '':
                    i += 1
                json_lines = []
                while i < len(lines) and lines[i].strip() != '':
                    json_lines.append(lines[i])
                    i += 1
                if not json_lines:
                    continue
                try:
                    arr = json.loads(''.join(json_lines))
                except Exception as e:
                    print(f"Failed to parse JSON for {key}: {e}")
                    continue

                if key not in sd:
                    print(f"Warning: parameter {key} not found in model; skipping")
                    continue

                target = sd[key]
                try:
                    tensor = torch.tensor(arr, dtype=target.dtype)
                except Exception:
                    # fallback to float32
                    tensor = torch.tensor(arr, dtype=torch.float32)

                try:
                    tensor = tensor.view(target.shape).to(target.device)
                except Exception as e:
                    print(f"Reshape/convert failed for {key}: {e}")
                    continue

                try:
                    target.copy_(tensor)
                except Exception as e:
                    print(f"Failed to copy parameter {key}: {e}")
            else:
                i += 1

        print(f"Loaded parameters from {path}")
    except Exception as e:
        print(f"Failed to load params: {e}")


# Attempt to load previously saved parameters (uncomment to use)
try:
    #ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    load_path = project_root / f"script_trained_params_final.txt"
    load_model_params(str(load_path))
except Exception as e:
    print(f"Warning: failed to load parameters after script training: {e}")
