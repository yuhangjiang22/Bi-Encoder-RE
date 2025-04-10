# Relation Extraction with Instance-Adapted Predicate Descriptions

Implementation for our RE approach, which employs a dual-encoder architecture with instance-adaptation techniques.

## Setup

### Install dependencies
Please install all the dependency packages using the following command:
```
pip install -r requirements.txt
```

### Decrypt datasets
We encrypted all datasets and provided with private keys. Use `decrypt.py` to decrypt each dataset saparately.

### Access to relation models
Please use the [link](https://drive.google.com/drive/folders/1VIIQkCkuokjkg766PE-Pn_gpCMnBqy90?usp=share_link) to download pre-trained models.

## Run relation models

```
python run_relation.py \
    --task {task} \
    --do_eval --eval_test \
    --model {model_path} \
    --do_lower_case \
    --context_window={context_window} \
    --max_seq_length={max_length} \
    --file_dir {file_folder} \
    --test_file "test.json" \
    --output_dir {output_dir}
```

## Train relation models
```
python 'run_relation.py' \
    --task {task} \
    --do_train --train_file {train_file} \
    --do_eval --eval_test \
    --model {model_path} \
    --do_lower_case \
    --output_dir {output_dir} \
    --eval_metric f1 \
    --train_batch_size={batch_size} \
    --eval_batch_size={batch_size} \
    --learning_rate={lr} \
    --num_train_epochs={num_epochs} \
    --context_window={context_window} \
    --max_seq_length={max_length} \
    --drop_out={drop_out} \
    --seed={seed} \
    --file_dir {file_dir} \
    --dev_file {dev_file} \
    --test_file {test_file}
```
