import torch
from torch import nn
import numpy as np
from typing import List
from transformers import PreTrainedModel, AutoConfig, AutoModel
from transformers import PretrainedConfig
from torch.nn import CrossEntropyLoss
class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


def contrastive_loss(
    scores: torch.FloatTensor,
    labels: List[int],
) -> torch.FloatTensor:
    log_softmax = torch.nn.functional.log_softmax
    batch_size, num_types = scores.size(0), scores.size(1)
    log_probs = log_softmax(scores, dim=-1)
    batch_indices = list(range(batch_size))
    log_probs = log_probs[batch_indices, labels]
    return - log_probs.mean()

BertLayerNorm = torch.nn.LayerNorm

class BEFREConfig(PretrainedConfig):

    def __init__(
        self,
        pretrained_model_name_or_path=None,
        cache_dir=None,
        use_auth_token=False,
        hidden_dropout_prob=0.1,
        max_span_width=30,
        use_span_width_embedding=False,
        linear_size=128,
        init_temperature=0.07,
        num_labels=6,
        prefix_projection=True,
        hidden_size=768,
        prefix_hidden_size=1024,
        pre_seq_len=60,
        num_hidden_layers=12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pretrained_model_name_or_path=pretrained_model_name_or_path
        self.cache_dir=cache_dir
        self.use_auth_token=use_auth_token
        self.hidden_dropout_prob=hidden_dropout_prob
        self.max_span_width = max_span_width
        self.use_span_width_embedding = use_span_width_embedding
        self.linear_size = linear_size
        self.init_temperature = init_temperature
        self.num_labels = num_labels
        self.prefix_projection = prefix_projection
        self.hidden_size = hidden_size
        self.prefix_hidden_size = prefix_hidden_size
        self.pre_seq_len = pre_seq_len
        self.num_hidden_layers = num_hidden_layers

class BEFRE(PreTrainedModel):

    config_class = BEFREConfig

    def __init__(self, config):
        super(BEFRE, self).__init__(config)

        hf_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path,
        )
        self.hf_config = hf_config
        self.config.pruned_heads = hf_config.pruned_heads
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(hf_config.hidden_size * 2)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.post_init()
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(hf_config.hidden_size * 2, config.num_labels)

        self.n_layer = hf_config.num_hidden_layers
        self.n_head = hf_config.num_attention_heads
        self.n_embd = hf_config.hidden_size // hf_config.num_attention_heads

        # self.prefix_encoders = {label: PrefixEncoder(config) for label in range(len(self.num_labels))}
        self.prefix_encoders = nn.ModuleList([
            PrefixEncoder(config) for i in range(self.num_labels)
        ])

        self.prefix_tokens = torch.arange(config.pre_seq_len).long()


        self.input_encoder = AutoModel.from_pretrained(
            config.pretrained_model_name_or_path,
            config=hf_config,
            add_pooling_layer=False
        )
        self.description_encoder = AutoModel.from_pretrained(
            config.pretrained_model_name_or_path,
            config=hf_config,
            add_pooling_layer=False
        )

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.hf_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.hf_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def gradient_checkpointing_enable(self):
        self.input_encoder.gradient_checkpointing_enable()
        self.description_encoder.gradient_checkpointing_enable()

    def get_prompt(self, batch_size, label):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.description_encoder.device)
        past_key_values = self.prefix_encoders[label](prefix_tokens)
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz,
            seqlen,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values


    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: torch.Tensor = None,
            token_type_ids: torch.Tensor = None,
            labels = None,
            sub_idx = None,
            obj_idx = None,
            descriptions_input_ids: torch.LongTensor = None,
            descriptions_input_mask: torch.Tensor = None,
            descriptions_type_ids: torch.Tensor = None,
            descriptions_sub_idx = None,
            descriptions_obj_idx = None,
            return_dict: bool = None,
    ):
        return_dict = return_dict if return_dict is not None else self.hf_config.use_return_dict
        batch_size, seq_length = input_ids.size()

        #reshape to size num_labels x batch_size x seq_len
        descriptions_input_ids = descriptions_input_ids.permute(1, 0, 2)
        descriptions_input_mask = descriptions_input_mask.permute(1, 0, 2)
        descriptions_type_ids = descriptions_type_ids.permute(1, 0, 2)

        past_key_values_for_each = []
        for a in range(self.num_labels):
            past_key_values_for_each.append(self.get_prompt(self, batch_size=batch_size, label=a))

        description_outputs = [self.description_encoder(
            descriptions_input_ids[i],
            attention_mask=descriptions_input_mask,
            token_type_ids=descriptions_type_ids,
            return_dict=return_dict,
            past_key_values=past_key_values_for_each[i]
        ) for i in range(self.num_labels)]
        description_outputs = torch.tensor(description_outputs)
        description_outputs = description_outputs.permute(1, 0, 2, 3)
        description_outputs = description_outputs.reshape(batch_size*self.num_labels, seq_length, self.hf_config.hidden_size)

        # description_outputs = self.description_encoder(
        #     descriptions_input_ids,
        #     attention_mask=descriptions_input_mask,
        #     token_type_ids=descriptions_type_ids,
        #     return_dict=return_dict,
        # )


        # batch_size*num_types x seq_length x hidden_size
        description_sequence_output = description_outputs

        outputs = self.input_encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
            return_dict=return_dict,
        )
        # batch_size x seq_length x hidden_size
        sequence_output = outputs[0]
        # batch_size*num_types x seq_length x hidden_size
        batch_size_times_num_types, des_seq_length, _ = description_sequence_output.size()
        num_types = int(batch_size_times_num_types / batch_size)

        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_idx)])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_idx)])
        rep = torch.cat((sub_output, obj_output), dim=1)
        rep = self.layer_norm(rep)

        # batch_size x hidden_size*2
        rep = self.dropout(rep)

        des_sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(description_sequence_output, descriptions_sub_idx)])
        des_obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(description_sequence_output, descriptions_obj_idx)])
        des_rep = torch.cat((des_sub_output, des_obj_output), dim=1)
        des_rep = self.layer_norm(des_rep)

        # batch_size*num_types x hidden_size*2
        des_rep = self.dropout(des_rep)

        results = []
        for i in range(batch_size):

            vec_input = rep[i]

            # Extract the corresponding num_types vectors from des_rep
            vec_des = des_rep[i * num_types:(i + 1) * num_types]

            # Calculate the dot product of each input rep with description rep
            cos = nn.CosineSimilarity(dim=-1)
            # dot_products = torch.matmul(vec_des, vec_input)
            dot_products = cos(vec_des, vec_input)
            results.append(dot_products)

        scores = torch.stack(results)
        scores = self.logit_scale.exp() * scores
        logits = self.classifier(rep)

        if labels is not None:
            CTloss = contrastive_loss(scores, labels)
            loss_fct = CrossEntropyLoss()
            CEloss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = 0.5 * CEloss + 0.5 * CTloss

            return loss
        else:
            return scores