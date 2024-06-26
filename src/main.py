# NLP project
# Detecting Bragging - Jin et al.
# Clare Treutel, Duncan Collins, Ryan Connolly
# Created: 3/15/24
# Updated: 4/13/24

# used some template text from the tutorials on these blogs:
# https://huggingface.co/learn/nlp-course/en/chapter3/3
# https://www.tensorflow.org/text/tutorials/classify_text_with_bert
# https://www.kaggle.com/code/harshjain123/bert-for-everyone-tutorial-implementation
# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb

# to make that annoying big "oneDNN custom operations are on. You may see slightly different numerical results"
# message that comes up at runtime go away
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import torch

import numpy as np
import csv
import evaluate
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
from datasets import load_dataset
from datasets import ClassLabel
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from numba import jit
from multiprocessing import pool
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time

tf.get_logger().setLevel('ERROR')

@jit(nopython=True)
def split_data(combined, keyword, random):
    """
    Split original bragging.csv file into test and train files
    keyword-sampled tweets for training, random-sampled for test
    remove non-essential columns, leave only text and label.
    input: file paths for combined, test and train data files
    output: n/a (creates test and train files)
    """
    with open(combined, 'r', newline='', encoding="utf-8") as combinedcsv, open(keyword, 'w+', newline='',
                                                                                encoding="utf-8") as train, open(random,
                                                                                                                 'w+',
                                                                                                                 newline='',
                                                                                                                 encoding="utf-8") as test:
        reader = csv.reader(combinedcsv)
        train_writer = csv.writer(train)
        test_writer = csv.writer(test)
        # specify text and label columns (removing everything else)
        cols = [1, 4]

        header = next(reader)
        train_writer.writerow(header[n] for n in cols)
        test_writer.writerow(header[n] for n in cols)

        for row in reader:
            sample_type = row[2]
            if sample_type == "keyword":
                train_writer.writerow(row[n] for n in cols)
            elif sample_type == "random":
                test_writer.writerow(row[n] for n in cols)


def get_baseline(test_df):
    """
    computes baseline metrics by assigning most frequent class to all predictions
    
    Input
    -----------------------------
    pandas df of test data
    
    Output
    -----------------------------
    accuracy, precision, recall metrics
    """

    X = test_df['text']
    Y = test_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    most_frequent_class = y_train.mode()[0]
    dummy_classifier = DummyClassifier(strategy='constant', constant=most_frequent_class)
    dummy_classifier.fit(X_train, y_train)
    y_pred = dummy_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0.0)
    recall = recall_score(y_test, y_pred, average='weighted')
    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall}


def preprocess(data):
    """
    Tokenize for BERT and convert str labels to ints

    Input
    -----------------------------
    batches of data from dataset
    
    Output
    -----------------------------
    batch of encoded dataset info
    """

    labels = ClassLabel(names_file = 'data/labels_binary.txt')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    tok = tokenizer(data['text'], padding='max_length')
    tok["label"] = labels.str2int(data['label'])
    return tok


def compute_metrics(eval_pred):
    """
    Computes metrics for model performance

    Input
    -----------------------------
    prediction/metrics object from trainer
    
    Output
    -----------------------------
    computed predictions
    """

    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1_score = evaluate.load("f1")
    # logits, labels = eval_pred
    logits = eval_pred[0]
    labels = eval_pred[1]
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels) | precision.compute(predictions=preds,
                                                                                      references=labels,
                                                                                      average="macro", zero_division=0.0) | recall.compute(
        predictions=preds, references=labels, average="macro") | f1_score.compute(predictions=preds, references=labels,
                                                                                  average="macro")


if __name__ == "__main__":
    modelType = "roberta-base"
    # replace with "bert-base-cased" for BERT
    # replace with "vinai/bertweet-base" for BERTweet
    bragging_data = 'data/bragging_data.csv'
    train = 'data/train_binary.csv'
    test = 'data/test_binary.csv'

    batch_size = 10
    learning_rate = .001
    num_epochs = 20

    # uncomment this to split data from original bragging_data.csv
    # split_data(bragging_data, train, test)

    dataset = load_dataset("csv", data_files={"train": [train], "test": [test]}).map(preprocess, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(modelType, num_labels=7).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    baseline_metrics = get_baseline(pd.read_csv(test))
    print("\nBaseline Metrics (majority class): \n" + str(baseline_metrics) + "\n")

    # This code is for generating the bar plots of label frequency (see images file)
    # train_label_counts = Counter(example['label'] for example in dataset['train'])
    # test_label_counts = Counter(example['label'] for example in dataset['test'])
    # This code makes bar plots of label frequencies in a dataset
    # labels = list(test_label_counts.keys())
    # frequencies = list(test_label_counts.values())
    # plt.figure(figsize=(10, 6))
    # plt.bar(labels, frequencies)
    # plt.xlabel('Labels')
    # plt.ylabel('Frequency')
    # plt.title('Testing Label Distribution')
    # plt.show()


    training_args = TrainingArguments(
        output_dir="test_trainer", 
        evaluation_strategy="epoch", 
        per_device_train_batch_size=batch_size, 
        fp16=True, 
        gradient_accumulation_steps=12
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler)
    )

    print("\n====================== TRAINING ======================\n")
    trainer.train()
    
    print("\n===================== PREDICTING =====================\n")
    predictions = trainer.predict(dataset["test"])
    preds = compute_metrics(predictions)
    # metric = evaluate.load()
    print("\nFinal predictions: \n" + str(preds) + "\n")

    pass
