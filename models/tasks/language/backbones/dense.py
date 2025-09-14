import torch 
import torch.nn as nn 
from models.components.attentions import load_attention, AttentionParameters
from models.components.base import FFN
from pydantic import BaseModel

class DenseModelParams(BaseModel):
  decoder_layers: int = 32

class DenseDecoderParams(BaseModel):
  input_dimension: int = 1024
  output_dimension : int  = 1024
  hidden_dimension: int = 1024
  norm_eps: float = 1e-5
  attention: AttentionParameters


class DenseTransformerParams(BaseModel):
  model: DenseModelParams
  decoder: DenseDecoderParams


class Decoder(nn.Module):
  def __init__(self, decoder_config: DenseDecoderParams):
    super().__init__()
    self.input_dimension = decoder_config.input_dimension
    self.output_dimension = decoder_config.output_dimension
    self.hidden_dimension = decoder_config.hidden_dimension

    self.attn = load_attention(decoder_config.attention)
    self.pre_norm = nn.RMSNorm(self.input_dimension, eps = decoder_config.norm_eps, elementwise_affine=True)
    self.post_attn_norm = nn.RMSNorm(self.input_dimension, eps = decoder_config.norm_eps, elementwise_affine=True)
    self.ffn = FFN(decoder_config.input_dimension, decoder_config.output_dimension, hidden_dimension = decoder_config.hidden_dimension)

  def forward(self, x: torch.Tensor):
    attn = self.attn(self.pre_norm(x))
    post_norm_sum = attn + x
    post_norm = self.post_attn_norm(post_norm_sum)
    ffn_output = self.ffn(post_norm)
    output = ffn_output + post_norm_sum
    return output

class DenseTransformer(nn.Module):
  def __init__(self, config: DenseTransformerParams):
    super().__init__()


    decoder_layers = []
    for i in range(config.model.decoder_layers):
      decoder_layers.append(Decoder(
        config.decoder
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