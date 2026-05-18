import torch
import torch.nn as nn
# from gidd.models.dit import LayerNorm, bias_dropout_add_scale_fused_train, modulate_fused
import torch.nn.functional as F
from einops import rearrange
from gidd.models.nvib_layer import NVIB 
import math
import typing

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
    nvib_alpha_tau: float = -45.0
    nvib_mu_tau: float = 1.0
    nvib_stdev_tau: float = 0.1
    nvib_learnable_prior: bool = False
    attention_dropout: float = 0.0



def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift


# @torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)


# @torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)



# @torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)

def rotate_half(x):
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_sep(x, cos, sin):
    """
    x: [B, H, S, D]
    cos/sin from Rotary()
    """

    # Original:
    # [1, S, 3, 1, D]

    cos = cos[:, :, 0, :, :]   # [1, S, 1, D]
    sin = sin[:, :, 0, :, :]   # [1, S, 1, D]

    # -> [1, 1, S, D]
    cos = cos.permute(0, 2, 1, 3)
    sin = sin.permute(0, 2, 1, 3)

    return (x * cos) + (rotate_half(x) * sin)

class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None,None,:]


class DDiT_NVIBBlock(nn.Module):
  def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
    super().__init__()
    self.n_heads = n_heads
    self.dim = dim
    self.cond_dim = cond_dim
    self.mlp_ratio = mlp_ratio

    self.n_kv_heads = n_heads
    self.head_dim = dim // n_heads
    self.scaling = self.head_dim ** -0.5

    self.norm1 = LayerNorm(dim)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()



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
    self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
    self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.attn_out = nn.Linear(n_heads * self.head_dim, dim, bias=False)
    self._last_nvib_outputs = None
    self.attention_dropout = config.attention_dropout

     # Store NVIB outputs so we can compute KL loss after forward pass
    self.kl_gaussian = None
    self.kl_dirichlet = None

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference


  def forward(self, x, rotary_cos_sin, c, seqlens=None, kl_loss=False):
    # print(self.training)
    batch_size, seq_len = x.shape[0], x.shape[1]

    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    (shift_msa, scale_msa, gate_msa, shift_mlp,
     scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

    # attention operation
    x_skip = x
    x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

    z, pi, mu, logvar, alpha, mask = self.nvib(
            encoder_output=x,
            batch_first=True,
            logging=self.training,
    )
    if self.training or kl_loss:
      self.kl_gaussian, self.kl_dirichlet = self.get_kl_loss(z, pi, mu, logvar, alpha, mask)



    query_states = self.q_proj(x)
    query_states = query_states.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
    
    input_for_kv = z # if self.training else mu

    key_states = self.k_proj(input_for_kv)
    value_states = self.v_proj(input_for_kv)
    
    key_states = key_states.view(batch_size, -1, self.n_kv_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, -1, self.n_kv_heads, self.head_dim).transpose(1, 2)
    
    cos, sin = rotary_cos_sin
      
    # Separate prior from sequence in keys
    k_prior = key_states[:, :, :1, :]   # (B, n_kv_heads, 1, head_dim) - Prior
    k_seq = key_states[:, :, 1:, :]     # (B, n_kv_heads, T, head_dim) - Sequence
    
    # print("k_prior shape:", k_prior.shape) #           torch.Size([2, 16, 1,   64])
    # print("k_seq shape:", k_seq.shape) #               torch.Size([2, 16, 512, 64])
    # print("query_states.shape:", query_states.shape) # torch.Size([2, 16, 512, 64])


    query_states = apply_rotary_pos_emb_sep(
        query_states,
        cos.to(query_states.dtype),
        sin.to(query_states.dtype),
    )

    k_seq = apply_rotary_pos_emb_sep(
        k_seq,
        cos.to(k_seq.dtype),
        sin.to(k_seq.dtype),
    )
      
    # Concatenate prior back: [prior, rotated_sequence]
    key_states = torch.cat([k_prior, k_seq], dim=2)  # (B, n_kv_heads, Nl, head_dim)
          
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
    
    pi_clamped = torch.clamp(pi, min=torch.finfo(pi.dtype).tiny)
    exp_scale = 1.0
    log_pi = torch.log(pi_clamped).permute(0, 2, 1).unsqueeze(1)  # (B, 1, 1, Nl)
    
    l2_norm = (torch.norm(input_for_kv, dim=-1, keepdim=True) ** 2)  # (B, Nl, 1)
    l2_norm = l2_norm.permute(0, 2, 1).unsqueeze(1)  # (B, 1, 1, Nl)
    scale_factor = 1.0 / (2.0 * math.sqrt(self.head_dim))

    attn_weights = attn_weights + log_pi - (scale_factor * l2_norm * exp_scale)

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()  # (B, T, n_heads, head_dim)
    attn_output = attn_output.view(batch_size, seq_len, self.dim)  # (B, T, C)


    x = bias_dropout_scale_fn(self.attn_out(attn_output),
                              None,
                              gate_msa,
                              x_skip,
                              self.dropout)

    # mlp operation
    x = bias_dropout_scale_fn(
      self.mlp(modulate_fused(
        self.norm2(x), shift_mlp, scale_mlp)),
      None, gate_mlp, x, self.dropout)
    return x


  def get_kl_loss(self, z, pi, mu, logvar, alpha, mask):
    # Compute KL divergence for Gaussian component
    # kl_gaussian expects: mu, logvar, alpha in (Nl, B, *) format
    kl_gaussian = self.nvib.kl_gaussian(
        mu=mu.transpose(0, 1),           # (Nl, B, C)
        logvar=logvar.transpose(0, 1),   # (Nl, B, C)
        alpha=alpha.transpose(0, 1),     # (Nl, B, 1)
        mask=mask,    # (B, Nl) - stays as is
    )  # Returns: (B,)
    
    # Compute KL divergence for Dirichlet component  
    # kl_dirichlet expects: alpha in (Nl, B, *) format
    kl_dirichlet = self.nvib.kl_dirichlet(
        alpha=alpha.transpose(0, 1),     # (Nl, B, 1)
        mask=mask,    # (B, Nl) - stays as is
    )  # Returns: (B,)
    
    return kl_gaussian, kl_dirichlet

  def get_kl_div(self) -> dict[str, torch.Tensor]:
    if self.kl_gaussian is None or self.kl_dirichlet is None:
        raise RuntimeError(
            "KL loss not available. Run forward() in training mode first to compute KL losses."
        )
    return self.kl_gaussian, self.kl_dirichlet


if __name__ == "__main__":
    block = DDiT_NVIBBlock(dim=1024, n_heads=16, cond_dim=128)

    # x.shape, rotary_cos_sin[0].shape, c.shape: torch.Size([1, 512, 1024]) torch.Size([1, 512, 3, 1, 64]) torch.Size([1, 128])

    x = torch.randn(2, 512, 1024)
    rotary_cos_sin = (
        torch.randn(1, 512, 3, 1, 64),
        torch.randn(1, 512, 3, 1, 64),
    )  
    c = torch.randn(2, 128)
    out = block(x, rotary_cos_sin, c, kl_loss=True)
    print("Forward pass successful.", out.shape)
    print("KL Gaussian:", block.kl_gaussian.shape)
    print("KL Dirichlet:", block.kl_dirichlet.shape)