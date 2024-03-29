#other implementation im working on, could be faster 



# NLP project
# Detecting Bragging - Jin et al.
# Clare Treutel, Duncan Collins, Ryan Connolly
# Created: 3/15/24
# Updated: 3/17/24

# used some template text from the tutorials on these blogs:
# https://huggingface.co/learn/nlp-course/en/chapter3/3
# https://www.tensorflow.org/text/tutorials/classify_text_with_bert
# https://www.kaggle.com/code/harshjain123/bert-for-everyone-tutorial-implementation
# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb

# to make that annoying big "oneDNN custom operations are on. You may see slightly different numerical results"
# message that comes up at runtime go away
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import csv
import evaluate
import tensorflow as tf
import torch.nn as nn
import torch.optim as optim

from datasets import load_dataset
from datasets import ClassLabel
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from numba import jit
from multiprocessing import pool
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR


import time

# import os
# import shutil
# import pandas as pd
import torch
# import tensorflow_hub as hub
# import tensorflow_text as text
# import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

@jit(nopython=True)
def split_data(combined, keyword, random):
    with open(combined, 'r', newline='', encoding="utf-8") as combinedcsv, open(keyword, 'w+', newline='', encoding="utf-8") as train, open(random, 'w+',
                                                                                                        newline='', encoding="utf-8") as test:
        reader = csv.reader(combinedcsv)
        train_writer = csv.writer(train)
        test_writer = csv.writer(test)
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

def preprocess(data):
    labels = ClassLabel(names_file='../data/labels.txt')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tok = tokenizer(data['text'], padding='max_length')
    tok["label"] = labels.str2int(data['label'])
    return tok

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1_score = evaluate.load("f1")
    # logits, labels = eval_pred
    logits = eval_pred[0]
    labels = eval_pred[1]
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels) | precision.compute(predictions=preds, references=labels, average="macro") | recall.compute(predictions=preds, references=labels, average="macro") | f1_score.compute(predictions=preds, references=labels, average="macro")


if __name__=="__main__":
    bragging_data = '../data/bragging_data.csv'
    train = '../data/train.csv'
    test = '../data/test.csv'

    batch_size = 8
    learning_rate = .1
    num_epochs = 20
    num_workers = 4

    #    uncomment this to split data from original bragging_data.csv
    # split_data(bragging_data, train, test)
    dataset = load_dataset("csv", data_files={"train": [train], "test": [test]}).map(preprocess, batched=True)
    labels = [label for label in dataset['train'].features.keys() if label not in ['text']]
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=7).to('cuda')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()


    for epoch in range(num_epochs): 
        model.train()
        for batch in train_loader: 
            optimizer.zero_grad()
            inputs, labels = torch.tensor(batch['text']).to('cuda'), torch.tensor(batch['labels']).to('cuda')
            outputs = model(**inputs)
            loss = criterion(outputs.logits, inputs['labels'].to('cuda'))
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad(): 
            total_correct = 0 
            total_samples = 0
            for batch in test_loader: 
                inputs, labels = torch.tensor(batch['text']).to('cuda'), torch.tensor(batch['labels']).to('cuda')
                outputs = model(**inputs)
                _, predicted = torch.max(outputs.logits, 1)
                total_correct += (predicted == inputs['labels'].to('cuda')).sum().item()
                total_samples += inputs['labels'].size(0)
            accuracy = total_correct / total_samples
            print(f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.4f}")