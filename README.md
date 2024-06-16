# Relation Extraction with Instance-Adapted Predicate Descriptions

This repo contains PyTorch code for our method, which employs a dual-encoder architecture with instance-adaptation techniques for relation extraction.

## Setup

### Install dependencies
Please install all the dependency packages using the following command:
```
pip install -r requirements.txt
```

### Access to pre-trained relation models
Please use the [link](https://drive.google.com/drive/folders/1VIIQkCkuokjkg766PE-Pn_gpCMnBqy90?usp=share_link) to download pre-trained models.

## Run the pre-trained relation model

```
python run_relation.py \
    --task chemprot_5 \
    --do_eval --eval_test \
    --model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --do_lower_case \
    --context_window=100 \
    --max_seq_length=250 \
    --file_dir "Bi-Encoder-RE/chemprot" \
    --test_file "test.json" \
    --output_dir "chemprt_5_model"
```

## Train/evaluate our model
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
    --seed=2024 \
    --file_dir {file_dir} \
    --dev_file {dev_file} \
    --test_file {test_file}
```