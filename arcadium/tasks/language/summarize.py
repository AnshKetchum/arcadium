import torch
from torchinfo import summary
from arcadium.models.moe import MoETransformerParams, MoEDecoderParams, MoEModelParams
from arcadium.tasks.language.loader import load_language_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = MoETransformerParams(
    model=MoEModelParams(),
    decoder=MoEDecoderParams()
)

name, model_type, net, tokenizer = load_language_model("configs/models/tiny-moe.yaml", device)

x = torch.zeros((1, 128), dtype=torch.long).to(device)
summary(net, input_data=x)
