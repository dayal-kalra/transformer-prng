import torch.nn as nn
import torch
from torch.nn import functional as F
import inspect
from dataclasses import dataclass
import numpy as np

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention layer.
    
    This implements the self-attention mechanism used in GPT models where each token
    can only attend to previous tokens in the sequence (causal masking).
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Combined projection for query, key, value vectors
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # Special flag for initialization scaling
        
        # Store dimensions
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dim
        
        # Project input into query, key, value vectors
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape to separate heads and transpose for attention computation
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        
        # Compute scaled dot-product attention with causal mask
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # Reshape back to original dimensions
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """Multi-layer perceptron used in transformer blocks.
    
    Contains two linear transformations with a GELU activation in between.
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)  # Expansion layer
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)  # Projection layer
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)    
        x = self.gelu(x)  
        x = self.c_proj(x)  
        return x


class Block(nn.Module):
    """Transformer block
    
    Each block contains a multi-head self-attention layer and an MLP,
    with layer normalization applied before each component.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Apply attention with residual connection
        x = x + self.attn(self.ln_1(x))
        # Apply MLP with residual connection
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT Model implementation.
    
    This implements the core GPT architecture with token embeddings,
    position embeddings, transformer blocks, and a model head.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),    # Token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),    # Position embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights between token embedding and LM head
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with scaled normal distribution."""
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """Forward pass through the model.
        
        Args:
            idx: Input token indices of shape (B, T)
            targets: Optional target tokens for computing loss
            
        Returns:
            tuple: (logits, loss) where loss is None if targets is None
        """
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # Get token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        
        # Combine embeddings and process through transformer blocks
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
            
        # Apply final layer norm and compute logits
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, beta1, beta2, device):
        """Configure optimizer with weight decay.
        
        Implements decoupled weight decay regularization by applying
        weight decay only to weight matrices and not to biases.
        """
        # Get parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Separate parameters into decay and no-decay groups
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        # Create optimizer groups
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # Print parameter counts
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # Use fused AdamW if available on CUDA
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=1e-8,
            fused=use_fused
        )
        return optimizer


class Abacus(nn.Module):
    def __init__(self, n_digit=2, embedding_dim=768):
        """
        embedding_dim (int): dimension to embed into
        """
        super().__init__()
        self.n_digit = n_digit
        self.embedding = torch.nn.Embedding(n_digit, embedding_dim)

    def forward(self, idx):
        B, T = idx.size()
        digit_ids = torch.tensor([i % self.n_digit for i in range(T)], device=idx.device)
        return self.embedding(digit_ids)


class GPT_oth_abacus(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(np.ceil(config.block_size / config.digits).astype(int), config.n_embd),
            abacus = nn.Embedding(config.digits, config.n_embd),
            # abacus = Abacus(config.digits, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()

        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos // self.config.digits) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        abacus_emb = self.transformer.abacus(pos % self.config.digits) # abacus embeddings of shape (T, n_embd)
        x = tok_emb + pos_emb + abacus_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    

    def configure_optimizers(self, weight_decay, learning_rate, beta1, beta2,device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2), eps=1e-8, fused=use_fused)
        return optimizer



@dataclass
class GPTConfig:
    """Configuration for GPT model."""
    block_size: int = 1024  # Maximum sequence length
    vocab_size: int = 50257  # GPT-2 vocabulary size (50,257 = 50,000 merges + 256 bytes + 1 special)
    n_layer: int = 12       # Number of transformer blocks
    n_head: int = 12        # Number of attention heads
    n_embd: int = 768      # Embedding dimension


@dataclass
class GPTConfig_abacus:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    digits: int = 2 # number of bits for the abacus encoding