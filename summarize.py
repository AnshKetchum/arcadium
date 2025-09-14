import torch 
from torchinfo import summary
from models.moe import MoETransformerParams, MoEDecoderParams, MoEModelParams
from models.loader import load_language_model

# Instantiate a representation of hardware
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = MoETransformerParams(
    model=MoEModelParams(),
    decoder=MoEDecoderParams() 
)

name, model_type, net, tokenizer = load_language_model("configs/models/tiny-moe.yaml", device )

# batch=1, seq_len=128, hidden_dim=1024
x = torch.zeros((1, 128), dtype=torch.long).to(device)
summary(net, input_data=x)  