Dataset

This project uses the Google Query Wellformedness (QWF) dataset:

Dataset repo:
https://github.com/google-research-datasets/query-wellformedness

Download the ZIP and extract it.
You will get:

train.tsv
dev.tsv
test.tsv
README.md

Required Folder Structure

The training script expects the dataset to be inside a folder named:

query-wellformedness-master


Your project folder must look like this:

project/
 ├── infer.py
 ├── qw_strong_weak_classifier.py
 ├── query-wellformedness-master/
 │     ├── train.tsv
 │     ├── dev.tsv
 │     ├── test.tsv
 │     └── README.md


Name must match exactly — otherwise the script will not find the dataset.

Training the Model

Run:

python3 qw_strong_weak_classifier.py


This will:

load the dataset

preprocess text

fine-tune DistilRoBERTa

tune the weak-query probability threshold

evaluate on dev/test

save the best model + tokenizer into:

output/distilroberta/

Running Inference

To classify a new query:

python3 infer.py --text "weather tomorrow"


Example result:

=== Query Classification ===
Text      : weather tomorrow
Prediction: WEAK
Weak prob : 0.9825
Threshold : 0.36
============================