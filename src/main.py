# used some template text from the tutorial on https://www.tensorflow.org/text/tutorials/classify_text_with_bert

import os
import shutil
import csv

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

        header = next(reader)
        train_writer.writerow(header)
        test_writer.writerow(header)

        for row in reader:
            sample_type = row[2]
            if sample_type == "keyword":
                train_writer.writerow(row)
            elif sample_type == "random":
                test_writer.writerow(row)

def main():
    bragging_data = '../data/bragging_data.csv'
    train = '../data/keyword.csv'
    test = '../data/random.csv'

    split_data(bragging_data, train, test)

main()