import json
import logging
import sys
import functools
import random
import os

from shared.data_structures import Dataset

logger = logging.getLogger('root')
CLS = "[CLS]"
SEP = "[SEP]"

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, sub_idx, obj_idx):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.sub_idx = sub_idx
        self.obj_idx = obj_idx

def decode_sample_id(sample_id):
    doc_sent = sample_id.split('::')[0]
    pair = sample_id.split('::')[1]
    pair = pair.split('-')
    sub = (int(pair[0][1:-1].split(',')[0]), int(pair[0][1:-1].split(',')[1]))
    obj = (int(pair[1][1:-1].split(',')[0]), int(pair[1][1:-1].split(',')[1]))

    return doc_sent, sub, obj


def generate_relation_data(entity_data, use_gold=False, context_window=0):
    """
    Prepare data for the relation model
    If training: set use_gold = True
    """
    logger.info('Generate relation data from %s' % (entity_data))
    data = Dataset(entity_data)
    directory = os.path.dirname(entity_data)
    json_file = os.path.join(directory, 'doc_key2doc.json')
    def read(json_file):
        with open(json_file, 'r') as f:
            key2doc = json.load(f)
        return  key2doc
    key2doc = read(json_file)
    nner, nrel = 0, 0
    max_sentsample = 0
    samples = []
    for doc in data:
        for i, sent in enumerate(doc):
            sent_samples = []

            nner += len(sent.ner)
            nrel += len(sent.relations)
            if use_gold:
                sent_ner = sent.ner
            else:
                sent_ner = sent.predicted_ner

            gold_ner = {}
            for ner in sent.ner:
                gold_ner[ner.span] = ner.label

            gold_rel = {}
            for rel in sent.relations:
                gold_rel[rel.pair] = rel.label

            sent_start = 0
            sent_end = len(sent.text)
            tokens = sent.text

            if context_window > 0:
                add_left = (context_window - len(sent.text)) // 2
                add_right = (context_window - len(sent.text)) - add_left

                j = i - 1
                while j >= 0 and add_left > 0:
                    context_to_add = doc[j].text[-add_left:]
                    tokens = context_to_add + tokens
                    add_left -= len(context_to_add)
                    sent_start += len(context_to_add)
                    sent_end += len(context_to_add)
                    j -= 1

                j = i + 1
                while j < len(doc) and add_right > 0:
                    context_to_add = doc[j].text[:add_right]
                    tokens = tokens + context_to_add
                    add_right -= len(context_to_add)
                    j += 1

            for x in range(len(sent_ner)):
                for y in range(len(sent_ner)):
                    if x == y:
                        continue
                    sub = sent_ner[x]
                    obj = sent_ner[y]
                    if sub.label == 'CHEMICAL' and obj.label == 'GENE':
                        label = gold_rel.get((sub.span, obj.span), 'no_relation')
                        if label == 'Other':
                            label = 'no_relation'
                        sample = {}
                        sample['docid'] = doc._doc_key
                        sample['id'] = '%s@%d::(%d,%d)-(%d,%d)' % (
                        doc._doc_key, sent.sentence_ix, sub.span.start_doc, sub.span.end_doc, obj.span.start_doc,
                        obj.span.end_doc)
                        sample['relation'] = label
                        sample['subj_start'] = sub.span.start_sent + sent_start
                        sample['subj_end'] = sub.span.end_sent + sent_start
                        sample['subj_type'] = sub.label
                        sample['obj_start'] = obj.span.start_sent + sent_start
                        sample['obj_end'] = obj.span.end_sent + sent_start
                        sample['obj_type'] = obj.label
                        sample['token'] = tokens
                        sample['sent_start'] = sent_start
                        sample['sent_end'] = sent_end
                        sample['doc_retrieved'] = key2doc[doc._doc_key]

                        sent_samples.append(sample)

            max_sentsample = max(max_sentsample, len(sent_samples))
            samples += sent_samples

    tot = len(samples)
    logger.info('#samples: %d, max #sent.samples: %d' % (tot, max_sentsample))

    return data, samples, nrel


def convert_examples_to_features(examples, label2id, max_seq_length, tokenizer, special_tokens, unused_tokens=True):
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

        SUBJECT_START_NER = get_special_token("SUBJ_START=%s"%example['subj_type'])
        SUBJECT_END_NER = get_special_token("SUBJ_END=%s"%example['subj_type'])
        OBJECT_START_NER = get_special_token("OBJ_START=%s"%example['obj_type'])
        OBJECT_END_NER = get_special_token("OBJ_END=%s"%example['obj_type'])

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
                tokens.append(SUBJECT_END_NER)
            if i == example['obj_end']:
                tokens.append(OBJECT_END_NER)
        tokens.append(SEP)
        document_retrieved = example['doc_retrieved']
        document_retrieved = document_retrieved['title'] + ' ' + document_retrieved['text']
        document_retrieved = document_retrieved.split()

        tokens = tokens + document_retrieved
        tokens.append(SEP)

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

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              sub_idx=sub_idx,
                              obj_idx=obj_idx))
    logger.info("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
    logger.info("Max #tokens: %d"%max_tokens)
    logger.info("%d (%.2f %%) examples can fit max_seq_length = %d" % (num_fit_examples,
                num_fit_examples * 100.0 / len(examples), max_seq_length))
    return features