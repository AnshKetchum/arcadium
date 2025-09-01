import torch 
import torch.nn as nn 
from models.components.base import Attention, FFN

class Decoder(nn.Module):
  def __init__(self, input_dimension, output_dimension, hidden_dimension = 512, num_heads = 8, norm_eps = 1e-5):
    super().__init__()
    self.input_dimension = input_dimension
    self.output_dimension = output_dimension
    self.hidden_dimension = hidden_dimension

    self.attn = Attention(num_heads, input_dimension)
    self.pre_norm = nn.RMSNorm(self.input_dimension, eps = norm_eps, elementwise_affine=True)
    self.post_attn_norm = nn.RMSNorm(self.input_dimension, eps = norm_eps, elementwise_affine=True)
    self.ffn = FFN(input_dimension, output_dimension, hidden_dimension = hidden_dimension)

  def forward(self, x: torch.Tensor):
    attn = self.attn(self.pre_norm(x))
    post_norm_sum = attn + x
    post_norm = self.post_attn_norm(post_norm_sum)
    ffn_output = self.ffn(post_norm)
    output = ffn_output + post_norm_sum
    return output

class DenseTransformer(nn.Module):
  def __init__(self, input_dimension, output_dimension, num_decoder_layers = 32, hidden_dimension = 512, num_heads = 8, norm_eps = 1e-5):
    super().__init__()


    decoder_layers = []
    for i in range(num_decoder_layers):
      decoder_layers.append(Decoder(
          input_dimension,
          output_dimension,
          hidden_dimension,
          num_heads,
          norm_eps
      ))

    self.decoder_layers = nn.ModuleList(decoder_layers)
    self.final_norm = nn.RMSNorm(input_dimension, eps = norm_eps, elementwise_affine=True)
    self.ffn = nn.Linear(input_dimension, output_dimension, bias = False)

  def forward(self, x):
    o = x

    for i, l in enumerate(self.decoder_layers):
        o = l(o)

    o = self.final_norm(o)
    o = self.ffn(o)
    return o