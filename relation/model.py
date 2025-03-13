import torch
from torch import nn
import numpy as np
from typing import List
from transformers import PreTrainedModel, AutoConfig, AutoModel
from transformers import PretrainedConfig
from torch.nn import CrossEntropyLoss


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
        alpha=0.5,
        tokenizer_len=None,
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
        self.alpha = alpha
        self.tokenizer_len = tokenizer_len

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
        self.layer_norm = BertLayerNorm(hf_config.hidden_size)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.post_init()
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(hf_config.hidden_size, config.num_labels)
        self.input_linear = nn.Linear(hf_config.hidden_size * 2, hf_config.hidden_size)
        self.des_linear = nn.Linear(hf_config.hidden_size * 3, hf_config.hidden_size)
        self.alpha = config.alpha
        self.tokenizer_len = config.tokenizer_len

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
        if self.tokenizer_len:
            self.input_encoder.resize_token_embeddings(self.tokenizer_len)
            self.description_encoder.resize_token_embeddings(self.tokenizer_len)


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

        description_outputs = self.description_encoder(
            descriptions_input_ids,
            attention_mask=descriptions_input_mask,
            token_type_ids=descriptions_type_ids,
            return_dict=return_dict,
        )

        # batch_size*num_types x seq_length x hidden_size
        description_sequence_output = description_outputs[0]

        outputs = self.input_encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
            return_dict=return_dict,
        )
        # batch_size x seq_length x hidden_size
        sequence_output = outputs[0]
        batch_size, seq_length, _ = sequence_output.size()
        # batch_size*num_types x seq_length x hidden_size
        batch_size_times_num_types, des_seq_length, _ = description_sequence_output.size()
        num_types = int(batch_size_times_num_types / batch_size)

        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_idx)])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_idx)])
        rep = torch.cat((sub_output, obj_output), dim=1)

        rep = self.input_linear(rep)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)

        cls_rep = torch.cat([a[0].unsqueeze(0) for a in sequence_output])
        cls_rep = cls_rep.repeat_interleave(num_types, dim=0)

        des_sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(description_sequence_output, descriptions_sub_idx)])
        des_obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(description_sequence_output, descriptions_obj_idx)])
        des_rep = torch.cat((des_sub_output, des_obj_output), dim=1)
        des_rep = torch.cat((des_rep, cls_rep), dim=1)
        des_rep = self.des_linear(des_rep)

        des_rep = self.layer_norm(des_rep)
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
            loss = self.alpha * CEloss + (1 - self.alpha) * CTloss
            return loss
        else:
            return scores

class BEFRE2(PreTrainedModel):

    config_class = BEFREConfig

    def __init__(self, config):
        super(BEFRE2, self).__init__(config)

        hf_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path,
        )
        self.hf_config = hf_config
        self.config.pruned_heads = hf_config.pruned_heads
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(hf_config.hidden_size)
        self.layer_norm_for_label = BertLayerNorm(config.num_labels)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.post_init()
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(hf_config.hidden_size, config.num_labels)
        self.input_linear = nn.Linear(hf_config.hidden_size * 2, hf_config.hidden_size)
        self.des_linear = nn.Linear(hf_config.hidden_size * 3, hf_config.hidden_size)
        self.alpha = config.alpha
        self.tokenizer_len = config.tokenizer_len
        self.relevance_net = nn.Sequential(
            nn.Linear(self.config.num_labels, hf_config.hidden_size),
            nn.ReLU(),
            nn.Linear(hf_config.hidden_size, 1),
            nn.Sigmoid()  # Output relevance score between 0 and 1
        )

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
        if self.tokenizer_len:
            self.input_encoder.resize_token_embeddings(self.tokenizer_len)
            self.description_encoder.resize_token_embeddings(self.tokenizer_len)


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

        description_outputs = self.description_encoder(
            descriptions_input_ids,
            attention_mask=descriptions_input_mask,
            token_type_ids=descriptions_type_ids,
            return_dict=return_dict,
        )

        # batch_size*num_types x seq_length x hidden_size
        description_sequence_output = description_outputs[0]

        outputs = self.input_encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
            return_dict=return_dict,
        )
        # batch_size x seq_length x hidden_size
        sequence_output = outputs[0]
        batch_size, seq_length, _ = sequence_output.size()
        # batch_size*num_types x seq_length x hidden_size
        batch_size_times_num_types, des_seq_length, _ = description_sequence_output.size()
        num_types = int(batch_size_times_num_types / batch_size)

        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_idx)])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_idx)])
        rep = torch.cat((sub_output, obj_output), dim=1)

        rep = self.input_linear(rep)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)

        cls_rep = torch.cat([a[0].unsqueeze(0) for a in sequence_output])
        cls_rep = cls_rep.repeat_interleave(num_types, dim=0)

        des_sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(description_sequence_output, descriptions_sub_idx)])
        des_obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(description_sequence_output, descriptions_obj_idx)])
        des_rep = torch.cat((des_sub_output, des_obj_output), dim=1)
        des_rep = torch.cat((des_rep, cls_rep), dim=1)
        des_rep = self.des_linear(des_rep)

        des_rep = self.layer_norm(des_rep)
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
        scores = self.layer_norm_for_label(scores)
        logits = self.layer_norm_for_label(logits)
        dynamic_relevance_score = self.relevance_net(logits).squeeze(
            -1)
        predictors = dynamic_relevance_score.unsqueeze(1) * scores + (
                    1 - dynamic_relevance_score.unsqueeze(1)) * logits

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            combined_loss = loss_fct(predictors.view(-1, self.num_labels), labels.view(-1))
            input_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return (combined_loss + input_loss) / 2
        else:
            return predictors

import torch
import torch.nn as nn

class MultiHeadDescriptionAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadDescriptionAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by the number of heads"

        # Linear layers for query, key, and value
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # Output linear layer to combine all heads
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attention_dropout = nn.Dropout(0.1)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, head_dim).
        Transpose the result so the shape is (batch_size, num_heads, seq_len, head_dim)
        """
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query_tensor, key_value_tensor, doc_mask=None):
        batch_size = query_tensor.size(0)
        doc_seq_len = key_value_tensor.size(1)

        # Linear projections
        query_layer = self.split_heads(self.query(query_tensor), batch_size)  # (batch_size, num_heads, 1, head_dim)
        key_layer = self.split_heads(self.key(key_value_tensor), batch_size)  # (batch_size, num_heads, doc_seq_len, head_dim)
        value_layer = self.split_heads(self.value(key_value_tensor), batch_size)  # (batch_size, num_heads, doc_seq_len, head_dim)

        # Compute scaled dot-product attention for each head
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / (self.head_dim ** 0.5)
        # (batch_size, num_heads, 1, doc_seq_len)

        # Apply mask if provided
        if doc_mask is not None:
            doc_mask = doc_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, doc_seq_len)
            attention_scores = attention_scores.masked_fill(doc_mask == 0, -1e9)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # (batch_size, num_heads, 1, doc_seq_len)
        attention_probs = self.attention_dropout(attention_probs)

        # Weighted sum of values
        attended_heads = torch.matmul(attention_probs, value_layer)  # (batch_size, num_heads, 1, head_dim)

        # Reshape back to (batch_size, seq_len, hidden_size) by concatenating the heads
        attended_heads = attended_heads.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        # Final linear layer to combine all heads
        attended_document_rep = self.out(attended_heads).squeeze(1)  # (batch_size, hidden_size)

        return attended_document_rep, attention_probs

class DualEncoder(PreTrainedModel):

    config_class = BEFREConfig

    def __init__(self, config):
        super(DualEncoder, self).__init__(config)

        hf_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path,
        )
        self.hf_config = hf_config
        self.config.pruned_heads = hf_config.pruned_heads
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(hf_config.hidden_size)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.post_init()
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(hf_config.hidden_size, config.num_labels)
        self.input_linear = nn.Linear(hf_config.hidden_size * 2, hf_config.hidden_size)
        self.des_linear = nn.Linear(hf_config.hidden_size * 3, hf_config.hidden_size)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.tokenizer_len = config.tokenizer_len
        self.description_attention = MultiHeadDescriptionAttention(hf_config.hidden_size, 12)

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
        if self.tokenizer_len:
            self.input_encoder.resize_token_embeddings(self.tokenizer_len)
            self.description_encoder.resize_token_embeddings(self.tokenizer_len)


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

        description_outputs = self.description_encoder(
            descriptions_input_ids,
            attention_mask=descriptions_input_mask,
            token_type_ids=descriptions_type_ids,
            return_dict=return_dict,
        )

        # batch_size*num_types x seq_length x hidden_size
        description_sequence_output = description_outputs[0]

        outputs = self.input_encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
            return_dict=return_dict,
        )
        # batch_size x seq_length x hidden_size
        sequence_output = outputs[0]
        batch_size, seq_length, _ = sequence_output.size()
        # batch_size*num_types x seq_length x hidden_size
        batch_size_times_num_types, des_seq_length, _ = description_sequence_output.size()
        num_types = int(batch_size_times_num_types / batch_size)

        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_idx)])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_idx)])
        rep = torch.cat((sub_output, obj_output), dim=1)

        rep = self.input_linear(rep)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        resized_rep = rep.repeat_interleave(num_types, dim=0)
        attended_des_rep, attention_probs = self.description_attention(resized_rep, description_sequence_output,
                                                                    doc_mask=descriptions_input_mask)

        # cls_rep = torch.cat([a[0].unsqueeze(0) for a in sequence_output])
        # cls_rep = cls_rep.repeat_interleave(num_types, dim=0)
        #
        #
        # des_sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(description_sequence_output, descriptions_sub_idx)])
        # des_obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(description_sequence_output, descriptions_obj_idx)])
        # des_rep = torch.cat((des_sub_output, des_obj_output), dim=1)
        # des_rep = torch.cat((des_rep, cls_rep), dim=1)
        # des_rep = self.des_linear(des_rep)

        # des_rep = self.layer_norm(des_rep)
        # des_rep = self.dropout(des_rep)

        des_rep = self.layer_norm(attended_des_rep)
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
            loss = self.alpha * CEloss + (1 - self.alpha) * CTloss
            return loss
        else:
            return scores