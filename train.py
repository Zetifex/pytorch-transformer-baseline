import torch
import torch.nn as nn
from torch.nn import functional as F
import urllib.request
from logger import ExperimentLogger

# -----------------------------------------------------------------------------
# 1. Setup, Configuration & Logging Pipeline
# -----------------------------------------------------------------------------
config = {
    "batch_size": 16,
    "block_size": 32,
    "max_iters": 3000,
    "eval_interval": 500,
    "learning_rate": 1e-3,
    "n_embd": 64,
    "n_head": 4,
    "n_layer": 4,
    "dropout": 0.1
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the custom JSON logger
logger = ExperimentLogger(config)

# -----------------------------------------------------------------------------
# 2. Data Pipeline & Tokenizer
# -----------------------------------------------------------------------------
print("Downloading Tiny Shakespeare dataset...")
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = urllib.request.urlopen(url).read().decode('utf-8')

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - config["block_size"], (config["batch_size"],))
    x = torch.stack([data_source[i:i+config["block_size"]] for i in ix])
    y = torch.stack([data_source[i+1:i+config["block_size"]+1] for i in ix])
    return x.to(device), y.to(device)

# -----------------------------------------------------------------------------
# 3. Transformer Architecture
# -----------------------------------------------------------------------------
class SingleHead(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)      
        q = self.query(x)    
        wei = q @ k.transpose(-2, -1) * (1.0 / (k.shape[-1] ** 0.5))
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)    
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([SingleHead(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
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
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

# -----------------------------------------------------------------------------
# 4. Execution & Inference
# -----------------------------------------------------------------------------
model = TransformerLanguageModel(
    vocab_size, 
    config["n_embd"], 
    config["n_head"], 
    config["n_layer"], 
    config["block_size"], 
    config["dropout"]
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

# Log metadata
param_count = sum(p.numel() for p in model.parameters())
logger.log_model_metadata(param_count, device)
print(f"Model parameters: {param_count/1e6:.2f} M")
print(f"Training on {device}...")

for iter in range(config["max_iters"]):
    xb, yb = get_batch('train')
    logits, loss = model(xb, targets=yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if iter % config["eval_interval"] == 0 or iter == config["max_iters"] - 1:
        current_loss = loss.item()
        print(f"Step {iter}: Loss {current_loss:.4f}")
        # Log the step to our JSON file
        logger.log_step(iter, current_loss)

print("\n--- Final Generation Output ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=400)[0].tolist()))