import torch
from torchvision.models.googlenet import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_full = googlenet(aux_logits=False).to(device)
model_to_quantify = googlenet(aux_logits=False).to(device)

# print(type(model_full) is torchvision.models.googlenet.GoogLeNet)
# print(isinstance(model_full, "torchvision.models.googlenet.GoogLeNet"))