"""
This code is based on the file in PURE repo: https://github.com/princeton-nlp/PURE/blob/main/run_relation.py
"""

import argparse
import logging
import os
import random
import time
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
# from collections import Counter
#
# from torch.nn import CrossEntropyLoss

from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
# from relation.models import BertForRelation, AlbertForRelation
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from relation.utils import generate_relation_data, decode_sample_id
from shared.const import task_rel_labels, task_ner_labels
# from relation.config import BEFREConfig
from relation.befre import BEFRE, BEFREConfig
from relation.unified_model import BEFRE, BEFREConfig

id2description = {0: ["there are no relations between the compound @subject@ and gene @object@ .",
                      "the compound @subject@ and gene @object@ has no relations ."],
                  1: ["the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an "
                      "upregulator , activator , or indirect upregulator in its interactions .",
                      "@subject@ initiates or enhances the activity of @object@ through direct or indirect means . an "
                      "upregulator ,activator , or indirect upregulator serves as the mechanism that increases the "
                      "function ,"
                      "expression , or activity of the @object@"
                      ],
                  2: ["the compound @subject@ has been identified to engage with the gene @object@ , manifesting as a "
                      "downregulator , inhibitor , or indirect downregulator in its interactions .",
                      "@subject@ interacts with the gene @object@ , resulting in a decrease in the gene's "
                      "activity or expression . This interaction can occur through direct inhibition , acting as a "
                      "downregulator , or through indirect means , where the compound causes a reduction in the gene's "
                      "function or expression without directly binding to it . Such mechanisms are crucial in "
                      "understanding genetic regulation and can have significant implications in fields like "
                      "pharmacology and gene therapy ."
                      ],
                  3: ["the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an "
                      "agonist , agonist activator , or agonist inhibitor in its interactions .",
                      "@subject@ interacts with the gene @object@ in a manner that modulates its activity positively ( "
                      "as an agonist or agonist activator ) or negatively ( as an agonist inhibitor ) . An agonist "
                      "interaction typically increases the gene's activity or the activity of proteins expressed by "
                      "the gene , whereas an agonist activator enhances this effect further . Conversely , an agonist "
                      "inhibitor would paradoxically bind in a manner that initially mimics an agonist's action but "
                      "ultimately inhibits the gene's activity or its downstream effects ."
                      ],
                  4: ["the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an "
                      "antagonist in its interactions .",
                      "@subject@ interacts with the gene @object@ by acting as an antagonist . This means that the "
                      "compound blocks or diminishes the gene's normal activity or the activity of the protein product "
                      "expressed by the gene . Antagonist interactions are significant in the regulation of biological "
                      "pathways and have wide-ranging implications in therapeutic interventions , where they can be "
                      "used to modulate the effects of genes involved in disease processes ."
                      ],
                  5: ["the compound @subject@ has been identified to engage with the gene @object@ , manifesting as a "
                      "substrate , product of, or substrate product of in its interactions .",
                      "@subject@ engages with the gene @object@ in a manner where it acts as a substrate , is a product"
                      "of, or both a substrate and product within the gene's associated biochemical pathways ."
                      ]}

# id2description = {0: "there are no relations between the compound @subject@ and gene @object@ .",
#                   1: "the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an "
#                      "upregulator, activator,or indirect upregulator in its interactions .",
#                   2: "the compound @subject@ has been identified to engage with thegene @object@ , manifesting as a "
#                      "downregulator, inhibitor, or indirect downregulator in its interactions .",
#                   3: "the compound @subject@ has been identified to engage with the gene @object@ , manifesting as an "
#                      "agonist, agonist activator, or agonist inhibitor in its interactions .",
#                   4: "the compound @subject@ has been identified toengage with the gene @object@ , manifesting as an "
#                      "antagonist in its interactions .",
#                   5: "the compound @subject@ hasbeen identified to engage with the gene @object@ , manifesting as a "
#                      "substrate, product of, or substrate product ofin its interactions ."}

# id2description = {0: "there are no relations between the compound @subject@ and gene @object@ .",
#                   1: "@subject@ engages @object@ , with upregulator , activator , or indirect upregulator .",
#                   2: "@subject@ is proved to be associated with @object@ , in downregulator , inhibitor , or indirect "
#                      "downregulator .",
#                   3: "@subject@ interacts with @object@ , in agonist , agonist activator , or agonist inhibitor .",
#                   4: "@subject@ is engaging @object@ , manifesting as an antagonist in the interactions .",
#                   5: "the compound @subject@ has been identified to interact with the gene @object@ , as a "
#                      "substrate , product of , or substrate product of in its interactions ."}

# id2description = {0: "there are no relations between the compound @subject@ and gene @object@ .",
#                   1: '@subject@ initiates or enhances the activity of @object@ through direct or indirect means . An '
#                      'upregulator ,'
#                      'activator , or indirect upregulator serves as the mechanism that increases the function , '
#                      'expression , or activity'
#                      'of the @object@',
#                   2: "@subject@ interacts with the gene @object@ , resulting in a decrease in the gene's "
#                      "activity or expression . This interaction can occur through direct inhibition , acting as a "
#                      "downregulator , or through indirect means , where the compound causes a reduction in the gene's "
#                      "function or expression without directly binding to it . Such mechanisms are crucial in "
#                      "understanding genetic regulation and can have significant implications in fields like "
#                      "pharmacology and gene therapy .",
#                   3: "@subject@ interacts with the gene @object@ in a manner that modulates its activity positively ( "
#                      "as an agonist or agonist activator ) or negatively ( as an agonist inhibitor ) . An agonist "
#                      "interaction typically increases the gene's activity or the activity of proteins expressed by "
#                      "the gene , whereas an agonist activator enhances this effect further . Conversely , an agonist "
#                      "inhibitor would paradoxically bind in a manner that initially mimics an agonist's action but "
#                      "ultimately inhibits the gene's activity or its downstream effects .",
#                   4: "@subject@ interacts with the gene @object@ by acting as an antagonist . This means that the "
#                      "compound blocks or diminishes the gene's normal activity or the activity of the protein product "
#                      "expressed by the gene . Antagonist interactions are significant in the regulation of biological "
#                      "pathways and have wide-ranging implications in therapeutic interventions , where they can be "
#                      "used to modulate the effects of genes involved in disease processes .",
#                   5: "@subject@ engages with the gene @object@ in a manner where it acts as a substrate , is a product "
#                      "of, or both a substrate and product within the gene's associated biochemical pathways ."}

tokenized_id2description = {key: [s.lower().split() for s in value] for key, value in id2description.items()}

def add_description_words(tokenizer, tokenized_id2description):
    unk_words = []
    for k, v in tokenized_id2description.items():
        for wds in v:
            for w in wds:
                if w not in tokenizer.vocab:
                    unk_words.append(w)
    tokenizer.add_tokens(unk_words)

CLS = "[CLS]"
SEP = "[SEP]"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 sub_idx,
                 obj_idx,
                 descriptions_input_ids,
                 descriptions_input_mask,
                 descriptions_type_ids,
                 descriptions_sub_idx,
                 descriptions_obj_idx):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.sub_idx = sub_idx
        self.obj_idx = obj_idx
        self.descriptions_input_ids = descriptions_input_ids
        self.descriptions_input_mask = descriptions_input_mask
        self.descriptions_type_ids = descriptions_type_ids
        self.descriptions_sub_idx = descriptions_sub_idx
        self.descriptions_obj_idx = descriptions_obj_idx


def add_marker_tokens(tokenizer, ner_labels):
    new_tokens = ['<SUBJ_START>', '<SUBJ_END>', '<OBJ_START>', '<OBJ_END>']
    for label in ner_labels:
        new_tokens.append('<SUBJ_START=%s>' % label)
        new_tokens.append('<SUBJ_END=%s>' % label)
        new_tokens.append('<OBJ_START=%s>' % label)
        new_tokens.append('<OBJ_END=%s>' % label)
    for label in ner_labels:
        new_tokens.append('<SUBJ=%s>' % label)
        new_tokens.append('<OBJ=%s>' % label)
    new_tokens = [token.lower() for token in new_tokens]
    tokenizer.add_tokens(new_tokens)
    logger.info('# vocab after adding markers: %d' % len(tokenizer))


# def convert_examples_to_features(examples, label2id, max_seq_length, tokenizer, special_tokens,
#                                  tokenized_id2description, unused_tokens=True):
#     """
#     Loads a data file into a list of `InputBatch`s.
#     unused_tokens: whether use [unused1] [unused2] as special tokens
#     """
#
#     def get_special_token(w):
#         if w not in special_tokens:
#             if unused_tokens:
#                 special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
#             else:
#                 special_tokens[w] = ('<' + w + '>').lower()
#         return special_tokens[w]
#
#     num_tokens = 0
#     max_tokens = 0
#     num_fit_examples = 0
#     num_shown_examples = 0
#     features = []
#     for (ex_index, example) in enumerate(examples):
#         if ex_index % 10000 == 0:
#             logger.info("Writing example %d of %d" % (ex_index, len(examples)))
#
#         tokens = [CLS]
#         SUBJECT_START = get_special_token("SUBJ_START")
#         SUBJECT_END = get_special_token("SUBJ_END")
#         OBJECT_START = get_special_token("OBJ_START")
#         OBJECT_END = get_special_token("OBJ_END")
#         SUBJECT_NER = get_special_token("SUBJ=%s" % example['subj_type'])
#         OBJECT_NER = get_special_token("OBJ=%s" % example['obj_type'])
#
#         SUBJECT_START_NER = get_special_token("SUBJ_START=%s" % example['subj_type'])
#         SUBJECT_END_NER = get_special_token("SUBJ_END=%s" % example['subj_type'])
#         OBJECT_START_NER = get_special_token("OBJ_START=%s" % example['obj_type'])
#         OBJECT_END_NER = get_special_token("OBJ_END=%s" % example['obj_type'])
#
#         for i, token in enumerate(example['token']):
#             if i == example['subj_start']:
#                 sub_idx = len(tokens)
#                 tokens.append(SUBJECT_START_NER)
#             if i == example['obj_start']:
#                 obj_idx = len(tokens)
#                 tokens.append(OBJECT_START_NER)
#             for sub_token in tokenizer.tokenize(token):
#                 tokens.append(sub_token)
#             if i == example['subj_end']:
#                 sub_idx_end = len(tokens)
#                 tokens.append(SUBJECT_END_NER)
#             if i == example['obj_end']:
#                 obj_idx_end = len(tokens)
#                 tokens.append(OBJECT_END_NER)
#         tokens.append(SEP)
#
#         subject = tokens[sub_idx:sub_idx_end + 1]
#         object = tokens[obj_idx:obj_idx_end + 1]
#
#         num_tokens += len(tokens)
#         max_tokens = max(max_tokens, len(tokens))
#
#         if len(tokens) > max_seq_length:
#             tokens = tokens[:max_seq_length]
#             if sub_idx >= max_seq_length:
#                 sub_idx = 0
#             if obj_idx >= max_seq_length:
#                 obj_idx = 0
#         else:
#             num_fit_examples += 1
#
#         segment_ids = [0] * len(tokens)
#         input_ids = tokenizer.convert_tokens_to_ids(tokens)
#         input_mask = [1] * len(input_ids)
#         padding = [0] * (max_seq_length - len(input_ids))
#         input_ids += padding
#         input_mask += padding
#         segment_ids += padding
#         label_id = label2id[example['relation']]
#
#         assert len(input_ids) == max_seq_length
#         assert len(input_mask) == max_seq_length
#         assert len(segment_ids) == max_seq_length
#
#         descriptions_input_ids = []
#         descriptions_input_mask = []
#         descriptions_type_ids = []
#         descriptions_sub_idx = []
#         descriptions_obj_idx = []
#
#         for _, description_tokens in tokenized_id2description.items():
#
#             description_tokens = [CLS] + description_tokens
#             description_tokens = [subject if word == '@subject@' else word for word in description_tokens]
#             description_tokens = [object if word == '@object@' else word for word in description_tokens]
#             description_tokens = [item for sublist in description_tokens for item in
#                                   (sublist if isinstance(sublist, list) else [sublist])]
#             description_tokens.append(SEP)
#
#             des_sub_idx = description_tokens.index(SUBJECT_START_NER)
#             des_obj_idx = description_tokens.index(OBJECT_START_NER)
#             descriptions_sub_idx.append(des_sub_idx)
#             descriptions_obj_idx.append(des_obj_idx)
#
#             if len(description_tokens) > max_seq_length:
#                 tokens = tokens[:max_seq_length]
#                 if sub_idx >= max_seq_length:
#                     sub_idx = 0
#                 if obj_idx >= max_seq_length:
#                     obj_idx = 0
#
#             description_input_ids = tokenizer.convert_tokens_to_ids(description_tokens)
#             description_type_ids = [0] * len(description_tokens)
#             description_input_mask = [1] * len(description_input_ids)
#             padding = [0] * (max_seq_length - len(description_input_ids))
#             description_input_ids += padding
#             description_input_mask += padding
#             description_type_ids += padding
#
#             assert len(description_input_ids) == max_seq_length
#             assert len(description_input_mask) == max_seq_length
#             assert len(description_type_ids) == max_seq_length
#
#             descriptions_input_ids.append(description_input_ids)
#             descriptions_input_mask.append(description_input_mask)
#             descriptions_type_ids.append(description_type_ids)
#
#         if num_shown_examples < 20:
#             if (ex_index < 5) or (label_id > 0):
#                 num_shown_examples += 1
#                 logger.info("*** Example ***")
#                 logger.info("guid: %s" % (example['id']))
#                 logger.info("tokens: %s" % " ".join(
#                     [str(x) for x in tokens]))
#                 logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
#                 logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
#                 logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
#                 logger.info("label: %s (id = %d)" % (example['relation'], label_id))
#                 logger.info("sub_idx, obj_idx: %d, %d" % (sub_idx, obj_idx))
#
#         features.append(
#             InputFeatures(input_ids=input_ids,
#                           input_mask=input_mask,
#                           segment_ids=segment_ids,
#                           label_id=label_id,
#                           sub_idx=sub_idx,
#                           obj_idx=obj_idx,
#                           descriptions_input_ids=descriptions_input_ids,
#                           descriptions_input_mask=descriptions_input_mask,
#                           descriptions_type_ids=descriptions_type_ids,
#                           descriptions_sub_idx=descriptions_sub_idx,
#                           descriptions_obj_idx=descriptions_obj_idx))
#     logger.info("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
#     logger.info("Max #tokens: %d" % max_tokens)
#     logger.info("%d (%.2f %%) examples can fit max_seq_length = %d" % (num_fit_examples,
#                                                                        num_fit_examples * 100.0 / len(examples),
#                                                                        max_seq_length))
#     return features

def convert_examples_to_features(examples, label2id, max_seq_length, tokenizer, special_tokens,
                                 tokenized_id2description, unused_tokens=False, multiple_descriptions=False):
    """
    Loads a data file into a list of `InputBatch`s.
    unused_tokens: whether use [unused1] [unused2] as special tokens
    """

    def get_special_token(w):
        if w not in special_tokens:
            if unused_tokens:
                special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
            else:
                special_tokens[w] = ('<' + w + '>').lower()
        return special_tokens[w]

    def get_description_input(description_tokens, des_max_seq_length=200):
        description_tokens = [CLS] + description_tokens
        description_tokens = [subject if word == '@subject@' else word for word in description_tokens]
        description_tokens = [object if word == '@object@' else word for word in description_tokens]
        description_tokens = [item for sublist in description_tokens for item in
                              (sublist if isinstance(sublist, list) else [sublist])]
        description_tokens.append(SEP)

        des_sub_idx = description_tokens.index(SUBJECT_START_NER)
        des_obj_idx = description_tokens.index(OBJECT_START_NER)
        descriptions_sub_idx.append(des_sub_idx)
        descriptions_obj_idx.append(des_obj_idx)

        description_input_ids = tokenizer.convert_tokens_to_ids(description_tokens)
        description_type_ids = [0] * len(description_tokens)
        description_input_mask = [1] * len(description_input_ids)
        padding = [0] * (des_max_seq_length - len(description_input_ids))
        description_input_ids += padding
        description_input_mask += padding
        description_type_ids += padding

        assert len(description_input_ids) == des_max_seq_length
        assert len(description_input_mask) == des_max_seq_length
        assert len(description_type_ids) == des_max_seq_length

        return description_input_ids, description_input_mask, description_type_ids

    num_tokens = 0
    max_tokens = 0
    num_fit_examples = 0
    num_shown_examples = 0
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = [CLS]
        SUBJECT_START = get_special_token("SUBJ_START")
        SUBJECT_END = get_special_token("SUBJ_END")
        OBJECT_START = get_special_token("OBJ_START")
        OBJECT_END = get_special_token("OBJ_END")
        SUBJECT_NER = get_special_token("SUBJ=%s" % example['subj_type'])
        OBJECT_NER = get_special_token("OBJ=%s" % example['obj_type'])

        SUBJECT_START_NER = get_special_token("SUBJ_START=%s" % example['subj_type'])
        SUBJECT_END_NER = get_special_token("SUBJ_END=%s" % example['subj_type'])
        OBJECT_START_NER = get_special_token("OBJ_START=%s" % example['obj_type'])
        OBJECT_END_NER = get_special_token("OBJ_END=%s" % example['obj_type'])

        for i, token in enumerate(example['token']):
            if i == example['subj_start']:
                sub_idx = len(tokens)
                tokens.append(SUBJECT_START_NER)
            if i == example['obj_start']:
                obj_idx = len(tokens)
                tokens.append(OBJECT_START_NER)
            for sub_token in tokenizer.tokenize(token):
                tokens.append(sub_token)
            if i == example['subj_end']:
                sub_idx_end = len(tokens)
                tokens.append(SUBJECT_END_NER)
            if i == example['obj_end']:
                obj_idx_end = len(tokens)
                tokens.append(OBJECT_END_NER)
        tokens.append(SEP)

        subject = tokens[sub_idx:sub_idx_end + 1]
        object = tokens[obj_idx:obj_idx_end + 1]

        num_tokens += len(tokens)
        max_tokens = max(max_tokens, len(tokens))

        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
            if sub_idx >= max_seq_length:
                sub_idx = 0
            if obj_idx >= max_seq_length:
                obj_idx = 0

        else:
            num_fit_examples += 1

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        label_id = label2id[example['relation']]

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        descriptions_input_ids = []
        descriptions_input_mask = []
        descriptions_type_ids = []
        descriptions_sub_idx = []
        descriptions_obj_idx = []

        if not multiple_descriptions:

            for _, description_tokens_list in tokenized_id2description.items():

                # description_tokens = random.choice(description_tokens_list)
                description_tokens = description_tokens_list[0]
                description_input_ids, description_input_mask, description_type_ids = get_description_input(description_tokens)

                descriptions_input_ids.append(description_input_ids)
                descriptions_input_mask.append(description_input_mask)
                descriptions_type_ids.append(description_type_ids)



        else:
            for label, description_tokens_list in tokenized_id2description.items():
                if label == label_id:
                    description_label_id = len(descriptions_input_ids)
                    description_tokens = description_tokens_list[0]
                    description_input_ids, description_input_mask, description_type_ids = get_description_input(
                        description_tokens)

                    descriptions_input_ids.append(description_input_ids)
                    descriptions_input_mask.append(description_input_mask)
                    descriptions_type_ids.append(description_type_ids)
                else:

                    for description_tokens in description_tokens_list:
                        description_input_ids, description_input_mask, description_type_ids = get_description_input(
                            description_tokens)

                        descriptions_input_ids.append(description_input_ids)
                        descriptions_input_mask.append(description_input_mask)
                        descriptions_type_ids.append(description_type_ids)




        if num_shown_examples < 20:
            if (ex_index < 5) or (label_id > 0):
                num_shown_examples += 1
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example['id']))
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label: %s (id = %d)" % (example['relation'], label_id))
                logger.info("sub_idx, obj_idx: %d, %d" % (sub_idx, obj_idx))

        if multiple_descriptions:
            label_id = description_label_id
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          sub_idx=sub_idx,
                          obj_idx=obj_idx,
                          descriptions_input_ids=descriptions_input_ids,
                          descriptions_input_mask=descriptions_input_mask,
                          descriptions_type_ids=descriptions_type_ids,
                          descriptions_sub_idx=descriptions_sub_idx,
                          descriptions_obj_idx=descriptions_obj_idx))
    logger.info("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
    logger.info("Max #tokens: %d" % max_tokens)
    logger.info("%d (%.2f %%) examples can fit max_seq_length = %d" % (num_fit_examples,
                                                                       num_fit_examples * 100.0 / len(examples),
                                                                       max_seq_length))
    return features

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_f1(preds, labels, e2e_ngold):
    n_gold = n_pred = n_correct = 0
    for pred, label in zip(preds, labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if (pred != 0) and (label != 0) and (pred == label):
            n_correct += 1
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0

        if e2e_ngold is not None:
            e2e_recall = n_correct * 1.0 / e2e_ngold
            e2e_f1 = 2.0 * prec * e2e_recall / (prec + e2e_recall)
        else:
            e2e_recall = e2e_f1 = 0.0
        return {'precision': prec, 'recall': e2e_recall, 'f1': e2e_f1, 'task_recall': recall, 'task_f1': f1,
                'n_correct': n_correct, 'n_pred': n_pred, 'n_gold': e2e_ngold, 'task_ngold': n_gold}


def evaluate(model, device, eval_dataloader, num_labels, eval_label_ids, batch_size, seq_len, e2e_ngold=None):
    model.eval()
    # eval_loss = 0
    nb_eval_steps = 0
    preds = []
    for input_ids, input_mask, segment_ids, label_ids, sub_idx, obj_idx, descriptions_input_ids, descriptions_input_mask, descriptions_type_ids, descriptions_sub_idx, descriptions_obj_idx in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        sub_idx = sub_idx.to(device)
        obj_idx = obj_idx.to(device)


        descriptions_input_ids = descriptions_input_ids.to(device)
        descriptions_input_mask = descriptions_input_mask.to(device)
        descriptions_type_ids = descriptions_type_ids.to(device)
        descriptions_sub_idx = descriptions_sub_idx.to(device)
        descriptions_obj_idx = descriptions_obj_idx.to(device)

        with torch.no_grad():
            scores = model(input_ids,
                           input_mask,
                           segment_ids,
                           labels=None,
                           sub_idx=sub_idx,
                           obj_idx=obj_idx,
                           descriptions_input_ids=descriptions_input_ids,
                           descriptions_input_mask=descriptions_input_mask,
                           descriptions_type_ids=descriptions_type_ids,
                           descriptions_sub_idx=descriptions_sub_idx,
                           descriptions_obj_idx=descriptions_obj_idx,
                           return_dict=True)

        # loss_fct = CrossEntropyLoss()
        # tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        # eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(scores.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], scores.detach().cpu().numpy(), axis=0)

    # eval_loss = eval_loss / nb_eval_steps
    # scores = preds[0]
    preds = np.argmax(preds[0], axis=1)
    result = compute_f1(preds, eval_label_ids.numpy(), e2e_ngold=e2e_ngold)
    result['accuracy'] = simple_accuracy(preds, eval_label_ids.numpy())
    # result['eval_loss'] = eval_loss

    return preds, result


def print_pred_json(eval_data, eval_examples, preds, id2label, output_file):
    rels = dict()
    for ex, pred in zip(eval_examples, preds):
        doc_sent, sub, obj = decode_sample_id(ex['id'])
        if doc_sent not in rels:
            rels[doc_sent] = []
        if pred != 0:
            rels[doc_sent].append([sub[0], sub[1], obj[0], obj[1], id2label[pred]])

    js = eval_data.js
    for doc in js:
        doc['predicted_relations'] = []
        for sid in range(len(doc['sentences'])):
            k = '%s@%d' % (doc['doc_key'], sid)
            doc['predicted_relations'].append(rels.get(k, []))

    logger.info('Output predictions to %s..' % (output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(doc) for doc in js))


def setseed(seed):
    random.seed(seed)
    np.random.seed(args.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_trained_model(output_dir, model, tokenizer):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    logger.info('Saving model to %s' % output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_vocabulary(output_dir)


def main(args):
    # if 'albert' in args.model:
    #     RelationModel = AlbertForRelation
    #     args.add_new_tokens = True
    # else:
    #     RelationModel = BertForRelation
    if args.train_befre:
        from relation.befre import BEFRE, BEFREConfig
    else:
        # from relation.testing_model import BEFRE, BEFREConfig
        from relation.prefix_model import BEFRE, BEFREConfig
        # from relation.unified_model import BEFRE, BEFREConfig
        # from relation.uni_model import BEFRE, BEFREConfig

    config = BEFREConfig(
        pretrained_model_name_or_path=args.model,
        cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE),
        revision=None,
        use_auth_token=True,
        hidden_dropout_prob=args.drop_out,
        prefix_projection=True,
        prefix_hidden_size=args.prefix_hidden_size,
        pre_seq_len=args.pre_seq_len,
    )
    setseed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    # train set
    if args.do_train:
        train_dataset, train_examples, train_nrel = generate_relation_data(args.train_file, use_gold=True,
                                                                           context_window=args.context_window)
    # dev set
    if (args.do_eval and args.do_train) or (args.do_eval and not (args.eval_test)):
        eval_dataset, eval_examples, eval_nrel = generate_relation_data(
            os.path.join(args.entity_output_dir, args.entity_predictions_dev), use_gold=args.eval_with_gold,
            context_window=args.context_window)
    # test set
    if args.eval_test:
        test_dataset, test_examples, test_nrel = generate_relation_data(
            os.path.join(args.entity_output_dir, args.entity_predictions_test), use_gold=args.eval_with_gold,
            context_window=args.context_window)


    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(sys.argv)
    logger.info(args)
    logger.info("device: {}, n_gpu: {}".format(
        device, n_gpu))

    # get label_list
    if os.path.exists(os.path.join(args.output_dir, 'label_list.json')):
        with open(os.path.join(args.output_dir, 'label_list.json'), 'r') as f:
            label_list = json.load(f)
    else:
        label_list = [args.negative_label] + task_rel_labels[args.task]
        with open(os.path.join(args.output_dir, 'label_list.json'), 'w') as f:
            json.dump(label_list, f)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
    add_description_words(tokenizer, tokenized_id2description)
    if args.add_new_tokens:
        add_marker_tokens(tokenizer, task_ner_labels[args.task])

    if os.path.exists(os.path.join(args.output_dir, 'special_tokens.json')):
        with open(os.path.join(args.output_dir, 'special_tokens.json'), 'r') as f:
            special_tokens = json.load(f)
    else:
        special_tokens = {}

    if args.do_eval and (args.do_train or not (args.eval_test)):
        eval_features = convert_examples_to_features(
            eval_examples, label2id, args.max_seq_length, tokenizer, special_tokens, tokenized_id2description,
            unused_tokens=not (args.add_new_tokens))
        logger.info("***** Dev *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_sub_idx = torch.tensor([f.sub_idx for f in eval_features], dtype=torch.long)
        all_obj_idx = torch.tensor([f.obj_idx for f in eval_features], dtype=torch.long)

        all_descriptions_input_ids = torch.tensor([f.descriptions_input_ids for f in eval_features],
                                                  dtype=torch.long)
        all_descriptions_input_mask = torch.tensor([f.descriptions_input_mask for f in eval_features],
                                                   dtype=torch.long)
        all_descriptions_type_ids = torch.tensor([f.descriptions_type_ids for f in eval_features],
                                                 dtype=torch.long)
        all_descriptions_sub_idx = torch.tensor([f.descriptions_sub_idx for f in eval_features], dtype=torch.long)
        all_descriptions_obj_idx = torch.tensor([f.descriptions_obj_idx for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids,
                                  all_input_mask,
                                  all_segment_ids,
                                  all_label_ids,
                                  all_sub_idx,
                                  all_obj_idx,
                                  all_descriptions_input_ids,
                                  all_descriptions_input_mask,
                                  all_descriptions_type_ids,
                                  all_descriptions_sub_idx,
                                  all_descriptions_obj_idx)

        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
        eval_label_ids = all_label_ids
    with open(os.path.join(args.output_dir, 'special_tokens.json'), 'w') as f:
        json.dump(special_tokens, f)

    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label2id, args.max_seq_length, tokenizer, special_tokens, tokenized_id2description,
            unused_tokens=not (args.add_new_tokens), multiple_descriptions=args.multi_descriptions)
        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(train_features)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_sub_idx = torch.tensor([f.sub_idx for f in train_features], dtype=torch.long)
        all_obj_idx = torch.tensor([f.obj_idx for f in train_features], dtype=torch.long)

        all_descriptions_input_ids = torch.tensor([f.descriptions_input_ids for f in train_features],
                                                  dtype=torch.long)
        all_descriptions_input_mask = torch.tensor([f.descriptions_input_mask for f in train_features],
                                                   dtype=torch.long)
        all_descriptions_type_ids = torch.tensor([f.descriptions_type_ids for f in train_features],
                                                 dtype=torch.long)
        all_descriptions_sub_idx = torch.tensor([f.descriptions_sub_idx for f in train_features], dtype=torch.long)
        all_descriptions_obj_idx = torch.tensor([f.descriptions_obj_idx for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids,
                                   all_input_mask,
                                   all_segment_ids,
                                   all_label_ids,
                                   all_sub_idx,
                                   all_obj_idx,
                                   all_descriptions_input_ids,
                                   all_descriptions_input_mask,
                                   all_descriptions_type_ids,
                                   all_descriptions_sub_idx,
                                   all_descriptions_obj_idx)

        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]
        if args.train_num_examples:
            train_batches = train_batches[:args.train_num_examples]

        num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs

        logger.info("***** Training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        best_result = None
        eval_step = max(1, len(train_batches) // args.eval_per_epoch)

        lr = args.learning_rate
        model = BEFRE(config)
        # model = RelationModel.from_pretrained(
        #     args.model, cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE), num_rel_labels=num_labels)

        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=not (args.bertadam))
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    int(num_train_optimization_steps * args.warmup_proportion),
                                                    num_train_optimization_steps)

        start_time = time.time()
        global_step = 0
        tr_loss = 0
        nb_tr_examples = 0
        nb_tr_steps = 0
        for epoch in range(int(args.num_train_epochs)):
            model.train()
            logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
            if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                random.shuffle(train_batches)
            for step, batch in enumerate(train_batches):
                # batch_size, _ = batch[0].size()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, sub_idx, obj_idx, descriptions_input_ids, descriptions_input_mask, descriptions_type_ids, descriptions_sub_idx, descriptions_obj_idx = batch

                loss = model(input_ids, input_mask, segment_ids, label_ids, sub_idx, obj_idx, descriptions_input_ids,
                             descriptions_input_mask, descriptions_type_ids, descriptions_sub_idx, descriptions_obj_idx,
                             return_dict=True)
                if n_gpu > 1:
                    loss = loss.mean()

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if (step + 1) % eval_step == 0:
                    logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                        epoch, step + 1, len(train_batches),
                               time.time() - start_time, tr_loss / nb_tr_steps))
                    save_model = False
                    if args.do_eval:
                        preds, result = evaluate(model=model,
                                                 device=device,
                                                 eval_dataloader=eval_dataloader,
                                                 eval_label_ids=eval_label_ids,
                                                 num_labels=num_labels,
                                                 batch_size=args.eval_batch_size,
                                                 seq_len=args.max_seq_length,
                                                 e2e_ngold=eval_nrel,
                                                 )
                        model.train()
                        result['global_step'] = global_step
                        result['epoch'] = epoch
                        result['learning_rate'] = lr
                        result['batch_size'] = args.train_batch_size

                        if (best_result is None) or (result[args.eval_metric] > best_result[args.eval_metric]):
                            best_result = result
                            logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                        (args.eval_metric, str(lr), epoch, result[args.eval_metric] * 100.0))
                            save_trained_model(args.output_dir, model, tokenizer)

    evaluation_results = {}
    if args.do_eval:
        logger.info(special_tokens)
        if args.eval_test:
            eval_dataset = test_dataset
            eval_examples = test_examples
            eval_features = convert_examples_to_features(
                test_examples, label2id, args.max_seq_length, tokenizer, special_tokens, tokenized_id2description,
                unused_tokens=not (args.add_new_tokens))
            eval_nrel = test_nrel
            logger.info(special_tokens)
            logger.info("***** Test *****")
            logger.info("  Num examples = %d", len(test_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            all_sub_idx = torch.tensor([f.sub_idx for f in eval_features], dtype=torch.long)
            all_obj_idx = torch.tensor([f.obj_idx for f in eval_features], dtype=torch.long)

            all_descriptions_input_ids = torch.tensor([f.descriptions_input_ids for f in eval_features],
                                                      dtype=torch.long)
            all_descriptions_input_mask = torch.tensor([f.descriptions_input_mask for f in eval_features],
                                                       dtype=torch.long)
            all_descriptions_type_ids = torch.tensor([f.descriptions_type_ids for f in eval_features],
                                                     dtype=torch.long)
            all_descriptions_sub_idx = torch.tensor([f.descriptions_sub_idx for f in eval_features], dtype=torch.long)
            all_descriptions_obj_idx = torch.tensor([f.descriptions_obj_idx for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids,
                                      all_input_mask,
                                      all_segment_ids,
                                      all_label_ids,
                                      all_sub_idx,
                                      all_obj_idx,
                                      all_descriptions_input_ids,
                                      all_descriptions_input_mask,
                                      all_descriptions_type_ids,
                                      all_descriptions_sub_idx,
                                      all_descriptions_obj_idx)

            eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
            eval_label_ids = all_label_ids

        model = BEFRE.from_pretrained(args.output_dir)
        model.to(device)
        preds, result = evaluate(model=model,
                                 device=device,
                                 eval_dataloader=eval_dataloader,
                                 eval_label_ids=eval_label_ids,
                                 num_labels=num_labels,
                                 batch_size=args.eval_batch_size,
                                 seq_len=args.max_seq_length,
                                 e2e_ngold=eval_nrel,
                                 )

        logger.info('*** Evaluation Results ***')
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        print_pred_json(eval_dataset, eval_examples, preds, id2label,
                        os.path.join(args.output_dir, args.prediction_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_per_epoch", default=10, type=int,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--negative_label", default="no_relation", type=str)
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--train_file", default=None, type=str, help="The path of the training data.")
    parser.add_argument("--train_mode", type=str, default='random_sorted',
                        choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_test", action="store_true", help="Whether to evaluate on final test set.")
    parser.add_argument("--eval_with_gold", action="store_true",
                        help="Whether to evaluate the relation model with gold entities provided.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_metric", default="f1", type=str)
    parser.add_argument("--learning_rate", default=None, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=0,
                        help="random seed for initialization")
    parser.add_argument("--bertadam", action="store_true", help="If bertadam, then set correct_bias = False")

    parser.add_argument("--entity_output_dir", type=str, default=None,
                        help="The directory of the prediction files of the entity model")
    parser.add_argument("--entity_predictions_dev", type=str, default="dev.json",
                        help="The entity prediction file of the dev set")
    parser.add_argument("--entity_predictions_test", type=str, default="test.json",
                        help="The entity prediction file of the test set")

    parser.add_argument("--prediction_file", type=str, default="predictions.json",
                        help="The prediction filename for the relation model")

    parser.add_argument('--task', type=str, default=None, required=True,
                        choices=['ace04', 'ace05', 'scierc', 'chemprot_5'])
    parser.add_argument('--context_window', type=int, default=0)

    parser.add_argument('--add_new_tokens', action='store_true',
                        help="Whether to add new tokens as marker tokens instead of using [unusedX] tokens.")
    parser.add_argument('--train_num_examples', type=int, default=None,
                        help="How many training instances to train")
    parser.add_argument('--train_befre', action='store_true',
                        help="Train BEFRE or other models.")
    parser.add_argument('--drop_out', type=float, default=0.1,
                        help="hidden drop out rate.")
    parser.add_argument('--multi_descriptions', action='store_true',
                        help="Use multi-descriptions or not.")
    parser.add_argument('--prefix_hidden_size', type=int, default=1024,
                        help="Prefix encoder hidden size.")
    parser.add_argument('--pre_seq_len', type=int, default=60,
                        help="Prefix encoder sequence length.")
    args = parser.parse_args()
    main(args)
