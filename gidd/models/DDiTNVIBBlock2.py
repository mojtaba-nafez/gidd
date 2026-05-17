import torch
import torch.nn as nn
from gidd.models.dit import LayerNorm, bias_dropout_add_scale_fused_train, modulate_fused, apply_rotary_pos_emb
import torch.nn.functional as F
from einops import rearrange
from gidd.models.nvib_layer import NVIB
import math

try:
  import flash_attn
  import flash_attn.layers.rotary
  has_flash_attn = True
except ImportError:
  torch.backends.cuda.enable_flash_sdp(enabled=True)
  has_flash_attn = False

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


from dataclasses import dataclass


@dataclass
class Config:
    nvib_prior_mu: float = None
    nvib_prior_var: float = None
    nvib_prior_log_alpha: float = None
    nvib_prior_log_alpha_stdev: float = None

    # optional fields (safe defaults)
    nvib_delta: float = 0.0
    nvib_alpha_tau: float = -10.0
    nvib_mu_tau: float = 1.0
    nvib_stdev_tau: float = 0.1
    nvib_learnable_prior: bool = False


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class DDiTBlock(nn.Module):
  """
  DiT Block with NVIB + Denoising Attention
  
  Architecture:
    x → norm1 → NVIB → Denoising Attention → residual → norm2 → MLP → residual
  
  The NVIB layer compresses the sequence into a stochastic latent space (z, pi, mu, logvar, alpha),
  and the Denoising Attention attends to this latent space instead of the raw input.
  """
  
  def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1, use_nvib=True, n_kv_heads=None):
    super().__init__()
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads  # GQA support
    self.dim = dim
    self.cond_dim = cond_dim
    self.mlp_ratio = mlp_ratio
    self.use_nvib = use_nvib
    self.head_dim = dim // n_heads
    self.num_key_value_groups = n_heads // self.n_kv_heads
    self.scaling = self.head_dim ** -0.5

    # Layer norms
    self.norm1 = LayerNorm(dim)
    self.norm2 = LayerNorm(dim)
    
    # Dropout
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout
    self.attention_dropout = dropout

    # AdaLN modulation
    self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()

    if self.use_nvib:
      # ═══════════════════════════════════════════════════════════════
      # NVIB: Nonparametric Variational Information Bottleneck
      # ═══════════════════════════════════════════════════════════════
      config = Config()
      self.nvib = NVIB(
          size_in=dim,
          size_out=dim,
          prior_mu=config.nvib_prior_mu,
          prior_var=config.nvib_prior_var,
          prior_log_alpha=config.nvib_prior_log_alpha,
          prior_log_alpha_stdev=config.nvib_prior_log_alpha_stdev,
          delta=config.nvib_delta,
          nheads=n_heads,
          alpha_tau=config.nvib_alpha_tau,
          mu_tau=config.nvib_mu_tau,
          stdev_tau=config.nvib_stdev_tau,
          learnable_prior=config.nvib_learnable_prior,
      )
      
      # ═══════════════════════════════════════════════════════════════
      # Denoising Attention Projections
      # ═══════════════════════════════════════════════════════════════
      # Query projection (from hidden states)
      self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
      
      # Key/Value projections (from NVIB latent z)
      self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
      self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
      
      # Output projection
      self.attn_out = nn.Linear(n_heads * self.head_dim, dim, bias=False)
      
      # Store last NVIB outputs for KL loss computation
      self._last_nvib_outputs = None
      
    else:
      # ═══════════════════════════════════════════════════════════════
      # Standard Self-Attention (without NVIB)
      # ═══════════════════════════════════════════════════════════════
      self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
      self.attn_out = nn.Linear(dim, dim, bias=False)

    # ═══════════════════════════════════════════════════════════════
    # Feed-Forward Network (unchanged)
    # ═══════════════════════════════════════════════════════════════
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))

  def flops(self, seq_len=128):
    per_token_flops = 0
    per_token_flops += 2 * self.dim * 3 * self.dim  # attn_qkv
    per_token_flops += 2 * seq_len * self.dim  # softmax attention
    per_token_flops += 2 * self.dim * self.dim  # attn_out
    per_token_flops += 2 * 2 * self.dim * 4 * self.dim  # mlp
    flops = per_token_flops * seq_len
    flops += 2 * self.cond_dim * 6 * self.dim  # adaLN_modulation
    return flops

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def forward(self, x, rotary_cos_sin, c, seqlens=None):
    """
    Forward pass with NVIB + Denoising Attention
    
    Args:
        x: Input tensor (B, T, C)
        rotary_cos_sin: Tuple of (cos, sin) for RoPE
        c: Conditioning tensor (B, cond_dim)
        seqlens: Optional sequence lengths for variable-length sequences
        
    Returns:
        x: Output tensor (B, T, C)
    """
    batch_size, seq_len = x.shape[0], x.shape[1]

    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    # AdaLN modulation parameters
    (shift_msa, scale_msa, gate_msa, shift_mlp,
     scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

    # ══════════════════════════════════════════════════════════════════
    # ATTENTION PATH
    # ══════════════════════════════════════════════════════════════════
    x_skip = x

    # Apply layer norm with AdaLN modulation
    x_norm = modulate_fused(self.norm1(x), shift_msa, scale_msa)  # (B, T, C)

    if self.use_nvib:
      # ────────────────────────────────────────────────────────────────
      # NVIB PATH: Compress to latent space
      # ────────────────────────────────────────────────────────────────
      nvib_outputs = self.nvib(
          encoder_output=x_norm,     # (B, T, C)
          batch_first=True,          # CRITICAL: NVIB expects batch-first
          logging=self.training,     # Enable logging during training
      )
      
      # Store for KL loss computation
      if self.training:
          self._last_nvib_outputs = nvib_outputs
      
      # Extract NVIB outputs
      z = nvib_outputs["z"]          # (B, Nl, C) where Nl = T+1 (includes prior)
      pi = nvib_outputs["pi"]        # (B, Nl, 1)
      mu = nvib_outputs["mu"]        # (B, Nl, C)
      logvar = nvib_outputs["logvar"]  # (B, Nl, C)
      alpha = nvib_outputs["alpha"]   # (B, Nl, 1)
      mask = nvib_outputs["memory_key_padding_mask"]  # (B, Nl)
      
      # ────────────────────────────────────────────────────────────────
      # DENOISING ATTENTION
      # ────────────────────────────────────────────────────────────────
      # 1. Project Query from normalized hidden states (length T)
      query_states = self.q_proj(x_norm)
      query_states = query_states.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
      # query_states: (B, n_heads, T, head_dim)
      
      # 2. Project Key/Value from NVIB latent (length Nl = T+1)
      # Use z (sample) during training, mu (mean) during inference
      input_for_kv = z if self.training else mu
      
      key_states = self.k_proj(input_for_kv)
      value_states = self.v_proj(input_for_kv)
      
      key_states = key_states.view(batch_size, -1, self.n_kv_heads, self.head_dim).transpose(1, 2)
      value_states = value_states.view(batch_size, -1, self.n_kv_heads, self.head_dim).transpose(1, 2)
      # key_states, value_states: (B, n_kv_heads, Nl, head_dim)
      
      # 3. Apply RoPE (Rotary Position Embeddings)
      # CRITICAL: Separate prior (index 0) from sequence (indices 1:Nl)
      cos, sin = rotary_cos_sin
      
      # Separate prior from sequence in keys
      k_prior = key_states[:, :, :1, :]   # (B, n_kv_heads, 1, head_dim) - Prior
      k_seq = key_states[:, :, 1:, :]     # (B, n_kv_heads, T, head_dim) - Sequence
      
      # Apply RoPE to query and sequence keys (both length T)
      query_states = apply_rotary_pos_emb(
          query_states.unsqueeze(2),  # (B, n_heads, 1, T, head_dim) for compatibility
          cos.to(query_states.dtype),
          sin.to(query_states.dtype)
      ).squeeze(2)  # (B, n_heads, T, head_dim)
      
      k_seq = apply_rotary_pos_emb(
          k_seq.unsqueeze(2),         # (B, n_kv_heads, 1, T, head_dim)
          cos.to(k_seq.dtype),
          sin.to(k_seq.dtype)
      ).squeeze(2)  # (B, n_kv_heads, T, head_dim)
      
      # Concatenate prior back: [prior, rotated_sequence]
      key_states = torch.cat([k_prior, k_seq], dim=2)  # (B, n_kv_heads, Nl, head_dim)
      
      # 4. Repeat K/V for Grouped Query Attention (GQA)
      if self.num_key_value_groups > 1:
          key_states = repeat_kv(key_states, self.num_key_value_groups)
          value_states = repeat_kv(value_states, self.num_key_value_groups)
      # key_states, value_states: (B, n_heads, Nl, head_dim)
      
      # 5. Compute attention scores
      # Q: (B, n_heads, T, head_dim) @ K^T: (B, n_heads, head_dim, Nl)
      # → attn_weights: (B, n_heads, T, Nl)
      attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
      
      # 6. Inject NVIB bias terms
      # Log pi term: (B, Nl, 1) → (B, 1, 1, Nl)
      pi_clamped = torch.clamp(pi, min=torch.finfo(pi.dtype).tiny)
      log_pi = torch.log(pi_clamped).permute(0, 2, 1).unsqueeze(1)  # (B, 1, 1, Nl)
      
      # L2 norm term: ||z||^2 / (2 * sqrt(head_dim))
      l2_norm = (torch.norm(input_for_kv, dim=-1, keepdim=True) ** 2)  # (B, Nl, 1)
      l2_norm = l2_norm.permute(0, 2, 1).unsqueeze(1)  # (B, 1, 1, Nl)
      scale_factor = 1.0 / (2.0 * math.sqrt(self.head_dim))
      
      # Add NVIB bias to attention weights
      attn_weights = attn_weights + log_pi - (scale_factor * l2_norm)
      
      # 7. Apply causal mask (if using Flash Attention path)
      # For standard attention, you might not need explicit masking
      # since DiT typically uses bidirectional attention
      
      # 8. Softmax and dropout
      attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
      attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
      
      # 9. Apply attention to values
      # attn_weights: (B, n_heads, T, Nl) @ values: (B, n_heads, Nl, head_dim)
      # → attn_output: (B, n_heads, T, head_dim)
      attn_output = torch.matmul(attn_weights, value_states)
      
      # 10. Reshape and project output
      attn_output = attn_output.transpose(1, 2).contiguous()  # (B, T, n_heads, head_dim)
      attn_output = attn_output.view(batch_size, seq_len, self.dim)  # (B, T, C)
      x = self.attn_out(attn_output)
      
    else:
      # ────────────────────────────────────────────────────────────────
      # STANDARD SELF-ATTENTION PATH (without NVIB)
      # ────────────────────────────────────────────────────────────────
      qkv = self.attn_qkv(x_norm)
      qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
      
      # Apply RoPE
      cos, sin = rotary_cos_sin
      qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
      
      if has_flash_attn:
          # Flash Attention path
          qkv = rearrange(qkv, 'b s ... -> (b s) ...')
          if seqlens is None:
              cu_seqlens = torch.arange(
                  0, (batch_size + 1) * seq_len, step=seq_len,
                  dtype=torch.int32, device=qkv.device)
          else:
              cu_seqlens = seqlens.cumsum(-1)
          x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
              qkv, cu_seqlens, seq_len, 0., causal=False)
          x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
      else:
          # Standard attention path
          q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
          x = F.scaled_dot_product_attention(q, k, v)
          x = rearrange(x, 'b h s d -> b s (h d)', b=batch_size)
      
      x = self.attn_out(x)

    # Apply residual connection with gating
    x = bias_dropout_scale_fn(x, None, gate_msa, x_skip, self.dropout)

    # ══════════════════════════════════════════════════════════════════
    # MLP PATH (unchanged)
    # ══════════════════════════════════════════════════════════════════
    x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp, x, self.dropout)
    
    return x

  def get_kl_loss(self):
      """
      Compute KL divergence losses from NVIB.
      Call this after forward() during training.
      
      Returns:
          Tuple of (kl_gaussian, kl_dirichlet) each shape (B,)
      """
      if not self.use_nvib:
          return None, None
      
      if self._last_nvib_outputs is None:
          raise RuntimeError("No NVIB outputs available. Run forward() in training mode first.")
      
      mu = self._last_nvib_outputs["mu"]      # (B, Nl, C)
      logvar = self._last_nvib_outputs["logvar"]  # (B, Nl, C)
      alpha = self._last_nvib_outputs["alpha"]    # (B, Nl, 1)
      mask = self._last_nvib_outputs["memory_key_padding_mask"]  # (B, Nl)
      
      # NVIB's KL methods expect no transpose for batch-first format
      kl_gaussian = self.nvib.kl_gaussian(
          mu=mu,
          logvar=logvar,
          alpha=alpha,
          mask=mask,
      )
      
      kl_dirichlet = self.nvib.kl_dirichlet(
          alpha=alpha,
          mask=mask,
      )
      
      return kl_gaussian, kl_dirichlet


if __name__ == "__main__":
    # Test with NVIB
    block_nvib = DDiTBlock(dim=1024, n_heads=16, cond_dim=128, use_nvib=True)
    block_nvib.train()
    
    # Test inputs
    x = torch.randn(2, 512, 1024)
    rotary_cos_sin = (
        torch.randn(1, 512, 3, 1, 64),
        torch.randn(1, 512, 3, 1, 64),
    )
    c = torch.randn(2, 128)
    
    # Forward pass
    out = block_nvib(x, rotary_cos_sin, c)
    print(f"✓ NVIB forward pass successful. Output shape: {out.shape}")
    
    # Get KL losses
    kl_g, kl_d = block_nvib.get_kl_loss()
    print(f"✓ KL Gaussian: {kl_g.mean().item():.6f}")
    print(f"✓ KL Dirichlet: {kl_d.mean().item():.6f}")
    
    # Test without NVIB (standard attention)
    block_standard = DDiTBlock(dim=1024, n_heads=16, cond_dim=128, use_nvib=False)
    out_standard = block_standard(x, rotary_cos_sin, c)
    print(f"✓ Standard forward pass successful. Output shape: {out_standard.shape}")
    
    print("\n✅ All tests passed!")