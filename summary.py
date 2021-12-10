from torchsummary import summary
from model import DexiNed
import torch

model = DexiNed()
summary(model, (3, 240, 368))
