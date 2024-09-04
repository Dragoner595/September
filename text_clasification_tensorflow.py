import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub 
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
print('Everything imported')

data = pd.read_csv('https://archive.org/download/fine-tune-bert-tensorflow-train.csv/train.csv.zip',
                   compression = 'zip',low_memory = False)

print(data.tail(20))
# we split data set into training and test data sets with selecting stratify for purpose of not overfiting or underfeating 

train_df, remaining =train_test_split(data , random_state = 42 , train_size = 0.0075, stratify = data.target.values)

valid_df, test_df = train_test_split(remaining,random_state = 42 , train_size = 0.00075,stratify = remaining.target.values)

print(train_df, valid_df)

# slising data set into tesorflow DataSet slicer with two target values 
with tf.device('/cpu:0'):
    train_data = tf.data.Dataset.from_tensor_slices((train_df['question_text'].values, train_df['target'].values))
    validation_data = tf.data.Dataset.from_tensor_slices((valid_df['question_text'].values,valid_df['target'].values))

print("Second print ",train_data,validation_data)

# Printing row and label thath represent what kind of comment is that 
for text, label in train_data.take(1):
    print(text)
    print(label)

# hypper paramiter
label_list = [0,1] # label categories 
max_seq_length = 128 #maximum length of (token) input seqences 
train_batch_size = 32 

# geting Bert layer and tokenezation 

bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4", trainable=False)

# pasing vocabluary path for our tokenizer 
vocab_files = bert_layer.resolved_object.vocab_file.asset_path.numpy()
# we using uncase version 
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
# 
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", clean_up_tokenization_spaces=True)

print("model finsihed ")