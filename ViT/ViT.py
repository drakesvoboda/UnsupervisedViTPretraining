import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair 

from transformers import BertConfig
from transformers import BertConfig, PreTrainedModel
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

class ViTConfig(BertConfig):
    def __init__(self, input_channels=3, patch_size=(16,16), position_embed_shape=(7,7), num_classes=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_channels = input_channels
        self.patch_size = _pair(patch_size)
        self.position_embed_shape = _pair(position_embed_shape)
        self.num_classes = num_classes

class ViTEmbedding(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.position_embed_shape = config.position_embed_shape
        self.positions = nn.Parameter(torch.randn(1, config.hidden_size, *self.position_embed_shape))
        self.patch_embedding = nn.Linear(config.input_channels * self.patch_size[0] * self.patch_size[1], config.hidden_size)
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, config.hidden_size))

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

class ViT(PreTrainedModel):
    def __init__(self, config: ViTConfig, add_pooling_layer=True):
        super().__init__(config)
        self.config_class = ViTConfig
        self.config = config
        self.embedding = ViTEmbedding(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

    def forward(self, pixel_values, return_dict=None, *args, **kwargs):
        embedding_output = self.embedding(pixel_values)
        encoder_outputs = self.encoder(embedding_output)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class ViTForSequenceEncoding(ViT):
    def forward(self, pixel_values, *args, **kwargs):
        return super().forward(pixel_values, return_dict=False)[1]

class ViTForSequenceClassification(ViTForSequenceEncoding):
    def __init__(self, config: ViTConfig, add_pooling_layer=True):
        super().__init__(config)
        self.cls = nn.Linear(config.hidden_size, config.num_classes) 

    def forward(self, pixel_values, *args, **kwargs):
        encoded = super().forward(pixel_values)
        return self.cls(encoded)