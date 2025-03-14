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
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from relation.utils import generate_relation_data, decode_sample_id, convert_examples_to_features, convert_biored_examples_to_features
from shared.const import task_rel_labels, task_ner_labels
from shared.descriptions import descriptions
from relation.model import BEFRE, BEFREConfig, DualEncoder, BEFRE2



def add_description_words(tokenizer, tokenized_id2description):

    def add_words(d):
        if type(d) == dict:
            for k,v in d.items():
                add_words(v)
        else:
            for wds in d:
                for w in wds:
                    if w not in tokenizer.vocab:
                        unk_words.append(w)

    unk_words = []
    add_words(tokenized_id2description)
    tokenizer.add_tokens(unk_words)



CLS = "[CLS]"
SEP = "[SEP]"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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



def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_f1(preds, labels):
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
        return {'precision': prec, 'recall': recall, 'f1': f1,
                'n_correct': n_correct, 'n_pred': n_pred, 'n_gold': n_gold}


def evaluate(model, device, eval_dataloader, eval_label_ids):
    model.eval()
    nb_eval_steps = 0
    preds = []
    for input_ids, input_mask, segment_ids, label_ids, sub_idx, obj_idx, descriptions_input_ids, descriptions_input_mask, descriptions_type_ids, descriptions_sub_idx, descriptions_obj_idx in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        sub_idx = sub_idx.to(device)
        obj_idx = obj_idx.to(device)

        batch_size, num_labels, des_seq_length = descriptions_input_ids.size()
        descriptions_input_ids = descriptions_input_ids.reshape(batch_size * num_labels, des_seq_length)
        descriptions_input_mask = descriptions_input_mask.reshape(batch_size * num_labels, des_seq_length)
        descriptions_type_ids = descriptions_type_ids.reshape(batch_size * num_labels, des_seq_length)
        descriptions_sub_idx = descriptions_sub_idx.reshape(batch_size * num_labels)
        descriptions_obj_idx = descriptions_obj_idx.reshape(batch_size * num_labels)
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

        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(scores.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], scores.detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds[0], axis=1)
    result = compute_f1(preds, eval_label_ids.numpy())
    result['accuracy'] = simple_accuracy(preds, eval_label_ids.numpy())

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

    setseed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    # train set
    if args.do_train:
        train_dataset, train_examples, train_nrel = generate_relation_data(args.train_file, context_window=args.context_window, task=args.task)
    # dev set
    if (args.do_eval and args.do_train) or (args.do_eval and not (args.eval_test)):
        eval_dataset, eval_examples, eval_nrel = generate_relation_data(
            os.path.join(args.file_dir, args.dev_file),
            context_window=args.context_window, task=args.task)
    # test set
    if args.eval_test:
        test_dataset, test_examples, test_nrel = generate_relation_data(
            os.path.join(args.file_dir, args.test_file),
            context_window=args.context_window, task=args.task)


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

    config = BEFREConfig(
        pretrained_model_name_or_path=args.model,
        cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE),
        revision=None,
        use_auth_token=True,
        hidden_dropout_prob=args.drop_out,
        num_labels=num_labels,
        alpha=args.alpha,
    )
    id2description = descriptions[args.task]

    if args.task != 'biored':
        tokenized_id2description = {key: [s.lower().split() for s in value] for key, value in id2description.items()}
        convert_function = convert_examples_to_features
    else:
        tokenized_id2description = {pair: {key: [s.lower().split() for s in value] for key, value in dic.items()} for
                                    pair, dic in id2description.items()}
        convert_function = convert_biored_examples_to_features

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
        eval_features = convert_function(
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
        train_features = convert_function(
            train_examples, label2id, args.max_seq_length, tokenizer, special_tokens, tokenized_id2description,
            unused_tokens=not (args.add_new_tokens))
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

        if 'PubMedBERT' not in config.pretrained_model_name_or_path:
            config.tokenizer_len = len(tokenizer)

        model = BEFRE2(config)

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
                batch_size, num_labels, des_seq_length = descriptions_input_ids.size()
                descriptions_input_ids = descriptions_input_ids.reshape(batch_size * num_labels, des_seq_length)
                descriptions_input_mask = descriptions_input_mask.reshape(batch_size * num_labels, des_seq_length)
                descriptions_type_ids = descriptions_type_ids.reshape(batch_size * num_labels, des_seq_length)
                descriptions_sub_idx = descriptions_sub_idx.reshape(batch_size * num_labels)
                descriptions_obj_idx = descriptions_obj_idx.reshape(batch_size * num_labels)
                print(descriptions_input_ids.size(), descriptions_input_mask.size(), descriptions_type_ids.size())
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

    if args.do_eval:
        logger.info(special_tokens)
        if args.eval_test:
            eval_dataset = test_dataset
            eval_examples = test_examples
            eval_features = convert_function(
                test_examples, label2id, args.max_seq_length, tokenizer, special_tokens, tokenized_id2description,
                unused_tokens=not (args.add_new_tokens))
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

        model = BEFRE2.from_pretrained(args.output_dir, num_labels=num_labels)
        model.to(device)
        preds, result = evaluate(model=model,
                                 device=device,
                                 eval_dataloader=eval_dataloader,
                                 eval_label_ids=eval_label_ids,
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
    parser.add_argument("--file_dir", type=str, default=None,
                        help="The directory of the prediction files of the entity model")
    parser.add_argument("--dev_file", type=str, default="dev.json",
                        help="The entity prediction file of the dev set")
    parser.add_argument("--test_file", type=str, default="test.json",
                        help="The entity prediction file of the test set")
    parser.add_argument("--prediction_file", type=str, default="predictions.json",
                        help="The prediction filename for the relation model")
    parser.add_argument('--task', type=str, default=None, required=True,
                        choices=['scierc', 'chemprot', 'chemprot_5', 'biored'])
    parser.add_argument('--context_window', type=int, default=0)
    parser.add_argument('--add_new_tokens', action='store_true',
                        help="Whether to add new tokens as marker tokens instead of using [unusedX] tokens.")
    parser.add_argument('--train_num_examples', type=int, default=None,
                        help="Number of training instances.")
    parser.add_argument('--train_pure', action='store_true',
                        help="Train PURE of BEFRE.")
    parser.add_argument('--drop_out', type=float, default=0.1,
                        help="hidden drop out rate.")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="alpha value for loss function.")

    args = parser.parse_args()
    main(args)
