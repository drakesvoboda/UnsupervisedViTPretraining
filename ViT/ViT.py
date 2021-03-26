import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair 

import torchvision
import torchvision.transforms as transforms

from transformers import BertConfig, BertModel

class ViTEmbedding(nn.Module):
    def __init__(self, input_channels, embed_dim, patch_size, position_embed_shape):
        super().__init__()
        self.patch_size = _pair(patch_size)
        self.position_embed_shape = _pair(position_embed_shape)
        self.positions = nn.Parameter(torch.randn(1, embed_dim, *self.position_embed_shape))
        self.patch_embedding = nn.Linear(input_channels * self.patch_size[0] * self.patch_size[1], embed_dim)
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, im):
        b, c, h, w = im.shape
        im = F.pad(im, (0, 0, h%self.patch_size[0], w%self.patch_size[1]))
        patches = im.unfold(2, *self.patch_size).unfold(3, *self.patch_size).permute(0, 2, 3, 1, 4, 5).contiguous() 
        position_embeddings = F.interpolate(self.positions, patches.shape[1:3])

        patches = patches.flatten(start_dim=3).flatten(start_dim=1, end_dim=2)
        position_embeddings = position_embeddings.flatten(start_dim=2).permute(0, 2, 1)

        embeds = self.patch_embedding(patches) + position_embeddings

        b, s, d = embeds.shape
        cls_embedding = self.cls_embedding.repeat(b, 1, 1)
        embeds = torch.cat((cls_embedding, embeds), dim=1)

        return embeds

class ViT(nn.Module):
    def __init__(self, input_channels=3, hidden_size=768, patch_size=16, position_embed_shape=(7,7)):
        super().__init__()
        self.embedding = ViTEmbedding(input_channels, hidden_size, patch_size, position_embed_shape)

        self.config = BertConfig(
            intermediate_size=hidden_size*4,
            hidden_size=hidden_size,
            num_hidden_layers=8, 
            num_attention_heads=8
        )

        self.transformer = BertModel(self.config)

    def forward(self, im):
        return self.transformer(inputs_embeds=self.embedding(im))

class ViTForClassification(ViT):
    def __init__(self, num_classes, input_channels=3, hidden_size=768, patch_size=16, position_embed_shape=(7,7)): 
        super().__init__(input_channels, hidden_size, patch_size, position_embed_shape)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, im):
        encoded = super().forward(im)[1]
        encoded = self.dropout(encoded)
        return self.classifier(encoded)