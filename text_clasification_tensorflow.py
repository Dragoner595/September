import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub 
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
print('Everything iomported')

data = pd.read_csv('https://archive.org/download/fine-tune-bert-tensorflow-train.csv/train.csv.zip',
                   compression = 'zip',low_memory = False)

print(data.tail(20))

train_df, remaining =train_test_split(data, random_state = 42 , train_size = 0.0075, stratify = data.target.values)

valid_df = train_test_split(remaining,random_state = 42 , train_size = 0.00075,stratify = remaining.target.values)

print(train_df,valid_df)
