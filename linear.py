import torch.nn as nn
import torch.nn.functional as F 

class LinearModel(nn.Module):
	def __init__(self, embed_dim, num_class):
		super(LinearModel, self).__init__()
		self.fc = nn.Linear(embed_dim, num_class)
		self.act = F.sigmoid

	def forward(self, x):
		return self.act(self.fc(x))