import torch
from torch import nn
import numpy as np
from typing import List
from transformers import PreTrainedModel, AutoConfig, AutoModel
from transformers import PretrainedConfig
import torch.nn.functional as F


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
        init_temperature=0.07,
        num_labels=6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pretrained_model_name_or_path=pretrained_model_name_or_path
        self.cache_dir=cache_dir
        self.use_auth_token=use_auth_token
        self.hidden_dropout_prob=hidden_dropout_prob
        self.max_span_width = max_span_width
        self.use_span_width_embedding = use_span_width_embedding
        self.init_temperature = init_temperature
        self.num_labels = num_labels

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.

    Modified from https://github.com/HobbitLong/SupContrast.
    """

    def __init__(self, temperature=0.01):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embs, same_label_mask, no_label_mask):
        """Compute supervised contrastive loss (variant).

        Args:
            embs: (num_examples, hidden_dim)
            same_label_mask: (num_examples, num_examples)
            no_label_mask: (num_examples)

        Returns:
            A loss scalar
        """
        # compute similarity scores for embs
        sim = embs @ embs.T / self.temperature
        # for numerical stability
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # compute log-likelihood for each pair
        # ***unlike original supcon, do not include examples with the same label in the denominator***
        negs = torch.exp(sim) * ~(same_label_mask)
        denominator = negs.sum(axis=1, keepdim=True) + torch.exp(sim)
        # log(exp(x)) = x and log(x/y) = log(x) - log(y)
        log_prob = sim - torch.log(denominator)

        # compute mean of log-likelihood over all positive pairs for a query/entity
        # exclude self from positive pairs
        pos_pairs_mask = same_label_mask.fill_diagonal_(0)
        # only include examples in loss that have positive pairs and the class label is known
        include_in_loss = (pos_pairs_mask.sum(1) != 0) & (~no_label_mask).flatten()
        # we add ones to the denominator to avoid nans when there are no positive pairs
        mean_log_prob_pos = (pos_pairs_mask * log_prob).sum(1) / (
            pos_pairs_mask.sum(1) + (~include_in_loss).float()
        )
        mean_log_prob_pos = mean_log_prob_pos[include_in_loss]

        # return zero loss if there are no values to take average over
        if mean_log_prob_pos.shape[0] == 0:
            return torch.tensor(0)

        # scale loss by temperature (as done in supcon paper)
        loss = -1 * self.temperature * mean_log_prob_pos

        # average loss over all queries/entities that have at least one positive pair
        return loss.mean()

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
        self.temperature = config.init_temperature

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

            cos = nn.CosineSimilarity(dim=-1)
            dot_products = cos(vec_des, vec_input)
            results.append(dot_products)

        scores = torch.stack(results)
        scores = self.logit_scale.exp() * scores

        if labels is not None:

            loss = contrastive_loss(scores, labels)

            all_rep = torch.cat([rep, des_rep], dim=0)
            all_rep = F.normalize(all_rep, p=2, dim=-1)

            des_labels = torch.arange(num_types).repeat(batch_size).to(self.description_encoder.device)
            all_labels = torch.cat([labels, des_labels], dim=0).view(-1, 1)
            no_label_mask = all_labels == 0
            same_label_mask = torch.eq(all_labels, all_labels.T).bool()
            relation_loss = SupConLoss(temperature=self.temperature)(
                embs=all_rep,
                no_label_mask=no_label_mask,
                same_label_mask=same_label_mask,
            )
            total_loss = 0.2 * loss + 0.8 * relation_loss
            return total_loss
        else:
            return scores












