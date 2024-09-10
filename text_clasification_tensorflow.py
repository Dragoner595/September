import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Input, Dense
import numpy as np
import sys
from transformers import InputExample, InputFeatures 

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

# pasing vocabluary path for our tokenizer (amount of vocabluare in liverary wich will be used for tokenezation purpose )
vocab_files = bert_layer.resolved_object.vocab_file.asset_path.numpy()
# we using uncase version ( to make it not case sensative )
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
# we estantiation our tokenezation ( this will tokenize our setances to tokens )
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", clean_up_tokenization_spaces=True)

print("model finsihed ")
# example of tokenezation on sentance 
print(tokenizer.wordpiece_tokenizer.tokenize('hi , how are you doing?'))
# converting tokens to the to the numeric token ids  
print(tokenizer.convert_tokens_to_ids((tokenizer.wordpiece_tokenizer.tokenize('hi , how are you doing?'))))

# in this function we inserting or feeding our model with test example and providing labels as 0 , 1  with max seqense 128 
# Function: This function now uses Hugging Face's encode_plus to tokenize and convert text into features that BERT expects.
def to_feature(text, label, max_seq_length=max_seq_length, tokenizer=tokenizer):
    # Tokenize input text
    inputs = tokenizer.encode_plus(
        text.numpy().decode('utf-8'),
        None,
        add_special_tokens=True,
        max_length=max_seq_length,
        truncation=True,
        padding='max_length',
        return_token_type_ids=True
    )

    input_ids = inputs['input_ids']
    input_mask = inputs['attention_mask']
    segment_ids = inputs['token_type_ids']
    
    return input_ids, input_mask, segment_ids, label.numpy()

# Wrap the function for use with tf.data.Dataset
def to_feature_map(text, label):
    input_ids, input_mask, segment_ids, label_id = tf.py_function(
        func=to_feature,
        inp=[text, label],
        Tout=[tf.int32, tf.int32, tf.int32, tf.int32]
    )
    
    # Set shape for inputs
    input_ids.set_shape([max_seq_length])
    input_mask.set_shape([max_seq_length])
    segment_ids.set_shape([max_seq_length])
    label_id.set_shape([])

    # Create the dictionary for BERT input
    x = {
        'input_word_ids': input_ids,
        'input_mask': input_mask,
        'input_type_ids': segment_ids
    }
    
    return x, label_id

# Creating input pipelines
with tf.device('/cpu:0'):
    # Train data pipeline
    train_data = (train_data
        .map(to_feature_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .shuffle(1000)
        .batch(train_batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    # Validation data pipeline
    validation_data = (validation_data
        .map(to_feature_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(train_batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
# we cheking right does our changes appeered 
# train data spec 
print(train_data.element_spec)
# valid data spec 
print(validation_data.element_spec)
# before we prepera data for bers layers to be acsepted for feature usage 
# we also devide out data into 3 different types of outpust what deliver corect output from bert model 

#                                            Building clasification model 

def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)


model = create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

# Print model summary
model.summary()