# used some template text from the tutorial on these blogs:
# https://www.tensorflow.org/text/tutorials/classify_text_with_bert
# https://www.kaggle.com/code/harshjain123/bert-for-everyone-tutorial-implementation
# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb

import os
import shutil
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')


def split_data(combined, keyword, random):
    with open(combined, 'r', newline='') as combinedcsv, open(keyword, 'w+', newline='') as train, open(random, 'w+',
                                                                                                        newline='') as test:
        reader = csv.reader(combinedcsv)
        train_writer = csv.writer(train)
        test_writer = csv.writer(test)
        cols = [1,4]

        header = next(reader)
        train_writer.writerow(header[n] for n in cols)
        test_writer.writerow(header[n] for n in cols)

        for row in reader:
            sample_type = row[2]
            if sample_type == "keyword":
                train_writer.writerow(row[n] for n in cols)
            elif sample_type == "random":
                test_writer.writerow(row[n] for n in cols)

# def preprocess(data):
#   TODO: preprocess/tokenize dataset

def main():
    bragging_data = '../data/bragging_data.csv'
    train = '../data/train.csv'
    test = '../data/test.csv'

#    uncomment this to split data from original bragging_data.csv
#    split_data(bragging_data, train, test)

    dataset = load_dataset("csv",data_files={"train":[train],"test":[test]})
    labels = [label for label in dataset['train'].features.keys() if label not in ['text']]
    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    labels = [label for label in dataset['train'].features.keys() if label not in ['text']]
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}

    # TODO: call preprocess fn
    # TODO: define model
    # TODO: train model
    # TODO: evaluate

main()
