import numpy as np
import torch.nn as nn
import torch
from Any2Any.model.any2one import Block_Unit,Convertor,Attention,Generator as SingleGenerator
import torch.nn.functional as F

def append_cond(x, cond):
    mean = cond.unsqueeze(dim=2)
    std = cond.unsqueeze(dim=2)
    std = torch.sqrt(torch.abs(std))
    out = x * std + mean
    return out

class Cont_Encoder(nn.Module):
	def __init__(self, d_model=256):
		super(Cont_Encoder, self).__init__()
		self.conv_block0 = nn.Sequential(
			Block_Unit(80, d_model, nn.ReLU()),
			Block_Unit(d_model, 36),
		)
		self.attention_norm0 = nn.Sequential(
			Attention(36, 2),
			nn.ReLU(),
			nn.InstanceNorm1d(36)
		)

		self.conv_block1 = nn.Sequential(
			Block_Unit(36, d_model, nn.ReLU()),
			Block_Unit(d_model, d_model),
		)
		self.attention_norm1 = nn.Sequential(
			Attention(d_model, 8),
			nn.ReLU(),
			nn.InstanceNorm1d(d_model)
		)


	def forward(self, x):
		x = x.transpose(1, 2) # [B,dim,len]
		x = self.conv_block0(x)  # [B,dim,len]
		x = self.attention_norm0(x)
		x = self.conv_block1(x)  # [B,dim,len]
		x = self.attention_norm1(x)
		return x

	def remove_weight_norm(self):
		for l in self.conv_block0:
			l.remove_weight_norm()
		for l in self.conv_block1:
			l.remove_weight_norm()


class Generator(nn.Module):
	def __init__(self, bottle_neck = 128, d_model=512):
		super(Generator, self).__init__()
		self.pre_block0 = nn.Sequential(
			Block_Unit(256, 256, nn.ReLU()),
			Block_Unit(256, bottle_neck),
		)
		self.attention0 = Attention(bottle_neck, 4)
		self.pre_block1 = nn.Sequential(
			Block_Unit(bottle_neck, 256, nn.ReLU()),
			Block_Unit(256, d_model),
		)
		self.attention1 = Attention(d_model, 8)
		self.smoothers = nn.Sequential(
			Convertor(d_model, 8, 1024),
			Convertor(d_model, 8, 1024),
			Convertor(d_model, 8, 1024)
		)  # len,B,dim

		self.post_block = nn.Sequential(
			Block_Unit(d_model, d_model, nn.ReLU()),
			Block_Unit(d_model , d_model//2, nn.ReLU()),
			Block_Unit(d_model//2, d_model//4,nn.ReLU()),
			Block_Unit(d_model//4, 80,nn.ReLU()),
			Block_Unit(80, 80,nn.ReLU())
		)

	def forward(self, word_enc, speak_emb):
		x = append_cond(word_enc, speak_emb) # B,dim,len
		x = self.pre_block0(x)  # B, dim, len
		x = F.relu(self.attention0(x))
		x = self.pre_block1(x)  # B, dim, len
		x = F.relu(self.attention1(x))

		x = x.transpose(0, 1).transpose(0, 2)
		x = self.smoothers(x)  # len, B ,512
		x = x.transpose(0, 1).transpose(1, 2)
		x = F.relu(self.post_block(x))  # B,512,len
		x = x.transpose(1, 2)  # [B,len,80]
		return x

	def remove_weight_norm(self):
		for l in self.pre_block0:
			l.remove_weight_norm()
		for l in self.pre_block1:
			l.remove_weight_norm()
		for l in self.smoothers:
			l.remove_weight_norm()
		for l in self.post_block:
			l.remove_weight_norm()
		
		
		

		
class MagicModel(nn.Module):
	def __init__(self, d_model=256):
		super(MagicModel, self).__init__()
		self.any2one = SingleGenerator() # B,len,dim
		self.cont_encoder = Cont_Encoder() # B,len,dim
		self.generator = Generator() #
		
	def forward(self, spk_input, mel, mel_masks):
		word_emb = self.any2one(mel,mel_masks)
		word_emb = torch.clamp(word_emb,0,1)
		word_enc = self.cont_encoder(word_emb)
		x = self.generator(word_enc, spk_input)
		return x

	def remove_weight_norm(self):
		self.any2one.remove_weight_norm()
		self.cont_encoder.remove_weight_norm()
		self.generator.remove_weight_norm()


def count_parameters_in_M(model):
	return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if
	              "auxiliary" not in name) / 1e6

if __name__ == '__main__':
	spk_input = torch.randn([5, 256])
	word_emb = torch.randn([5, 137, 80])
	magic = MagicModel()
	magic.remove_weight_norm()
	print("model params:", count_parameters_in_M(magic))
	print("model params:", count_parameters_in_M(magic.cont_encoder))
	print("model params:", count_parameters_in_M(magic.any2one))
	cont_emb = magic.cont_encoder(word_emb)
	out = magic(spk_input, word_emb,None)
	print("out shape:", out[0].shape)
