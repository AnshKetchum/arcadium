import torch 
import torch.nn as nn 
import torch.nn.functional as F
from pydantic import BaseModel

from models.components.attentions import load_attention, AttentionParameters
from models.components.moe_layer import MoE

class MoEModelParams(BaseModel):
  decoder_layers: int = 32

class MoEDecoderParams(BaseModel):
  input_dimension: int = 1024
  output_dimension : int  = 1024
  hidden_dimension: int = 1024
  experts: int = 8 
  norm_eps: float = 1e-5
  top_k: int = 2
  attention: AttentionParameters


class MoETransformerParams(BaseModel):
  model: MoEModelParams
  decoder: MoEDecoderParams

class MoEDecoder(nn.Module):
  def __init__(self, decoder_config: MoEDecoderParams):
    super().__init__()
    self.input_dimension = decoder_config.input_dimension
    self.output_dimension = decoder_config.output_dimension
    self.hidden_dimension = decoder_config.hidden_dimension

    self.attn = load_attention(decoder_config.attention)

    self.pre_norm = nn.RMSNorm(self.input_dimension, eps = decoder_config.norm_eps, elementwise_affine=True)
    self.post_attn_norm = nn.RMSNorm(self.input_dimension, eps = decoder_config.norm_eps, elementwise_affine=True)
    self.ffn = MoE(decoder_config.input_dimension, decoder_config.output_dimension, hidden_dimension = decoder_config.hidden_dimension, num_experts= decoder_config.experts, k = decoder_config.top_k)

  def forward(self, x: torch.Tensor):
    initial_shape = x.shape
    
    pre_norm = self.pre_norm(x)
    attn = self.attn(pre_norm)
    post_norm_sum = attn + x
    post_norm = self.post_attn_norm(post_norm_sum)
    ffn_output = self.ffn(post_norm)
    output = ffn_output + post_norm_sum

    final_shape = output.shape
    assert initial_shape == final_shape
    return output

class MoETransformer(nn.Module):
  def __init__(self, config: MoETransformerParams):
    super().__init__()


    decoder_layers = []
    for i in range(config.model.decoder_layers):
      decoder_layers.append(MoEDecoder(
        decoder_config=config.decoder
      ))

    self.decoder_layers = nn.ModuleList(decoder_layers)
    self.final_norm = nn.RMSNorm(config.decoder.input_dimension, eps = config.decoder.norm_eps, elementwise_affine=True)
    self.ffn = nn.Linear(config.decoder.input_dimension, config.decoder.output_dimension, bias = False)

  def forward(self, x):
    o = x
    
    for i, l in enumerate(self.decoder_layers):
        o = l(o)

    o = self.final_norm(o)
    o = self.ffn(o)
    return o
  
  def metadata(self):
    data =  []
    for i, l in enumerate(self.decoder_layers):
      data.append(l.attn.metadata())
    return data
