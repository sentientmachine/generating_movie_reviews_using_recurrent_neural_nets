#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

#from tensorflow import keras
#I like manual keras from source: 
import keras
import os
import time
import random
from datetime import datetime, timedelta
from urllib.request import urlopen

from keras.utils import to_categorical as one_hot

#num iterations says how long to train
num_iterations = 3000    #6000 takes an hour to train and generates good text

#load the data file of positive movie reviews into main python3 memory:
#make everything lowercase:
#txt = open('positive_movie_reviews.txt', 'r').read().lower()
#txt = open('positive_movie_reviews_small.txt', 'r').read().lower()

txt = open('strata_abstracts.txt', 'r').read().lower()

#show number of characters we have to work with
print(len(txt)) 
print()
#show the first 1000 characters, a sampling of the first positive review:
print(txt[:1000])

chars = list(set(txt))

#convert the characters to their numeric equivalents because RNN's like integers, not characters
data = [chars.index(c) for c in txt]

print('Count of unique characters (i.e., features):', len(chars))




def get_next_batch(batch_size, time_steps, data):
    x_batch = np.zeros((batch_size, time_steps))
    y_batch = np.zeros((batch_size, time_steps))

    batch_ids = range(len(data) - time_steps - 1)
    batch_id = random.sample(batch_ids, batch_size)

    for t in range(time_steps):
        x_batch[:, t] = [data[i+t] for i in batch_id]
        y_batch[:, t] = [data[i+t+1] for i in batch_id]

    return x_batch, y_batch

get_next_batch(1, 5, data)



lstm_size = 256
time_steps = 150
batch_size = 50

def create_LSTM_model(num_lstm_cells, input_batch_size, time_steps):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(num_lstm_cells, 
                                batch_input_shape=(input_batch_size, time_steps, len(chars)), 
                                return_sequences=True, 
                                stateful=True))

    model.add(keras.layers.LSTM(num_lstm_cells,
              batch_input_shape=(input_batch_size, time_steps, len(chars)),
              return_sequences=True, 
              stateful=True))

    model.add(keras.layers.TimeDistributed(keras.layers.Dense(len(chars), activation='softmax')))

    return model





model_training = create_LSTM_model(lstm_size, batch_size, time_steps)
RMSoptimizer = keras.optimizers.RMSprop(lr=0.003)
model_training.compile(loss='categorical_crossentropy', optimizer=RMSoptimizer)
model_predicting = create_LSTM_model(lstm_size, 1, None)





def generate_text(seed, len_test_txt=500):
    global model_predicting
    # copy weights from trained model to predicting model
    trained_weights = model_training.get_weights()
    model_predicting.set_weights(trained_weights)

    seed = seed.lower()
    gen_str = seed

    # turn seed from letters to numbers so we can then one-hot encode it
    seed = [chars.index(c) for c in seed]
    for i in range(len_test_txt):
        # one hot encode the seed
        seed_oh = one_hot(seed, num_classes = len(chars))

        # reshape the seed into the shape the input layer of lstm needs
        seed_oh = np.reshape(seed_oh, newshape = (1, -1, len(chars)))

        #predicting
        char_probabilities = model_predicting.predict(seed_oh, verbose = 0)[0][0]
        np.nan_to_num(char_probabilities)
        pred_index = np.random.choice(range(len(chars)), p = char_probabilities)
        gen_str += chars[pred_index]
        #update seed to be the predicted character
        seed = pred_index

    model_predicting.reset_states()

    return gen_str




# fitting the model, depending on `num_iterations`, this can take a while
display_step = 50
start_time = time.time()

for i in range(num_iterations):
    # Get a random batch of training examples.
    x_batch, y_true_batch = get_next_batch(batch_size, time_steps, data)
    # we need to one-hot encode inputs and outputs
    x = one_hot(x_batch, num_classes = len(chars))
    y = one_hot(y_true_batch, num_classes = len(chars))

    # ---------------------- TRAIN -------------------------
    # optimize model
    history = model_training.fit(x, y, verbose = 0, batch_size=x.shape[0])
    model_training.reset_states()

    # Print status every display_step iterations.
    if (i % display_step == 0) or (i == num_iterations - 1):
        #Message for network evaluation
        msg = "Optimization Iteration: {}, Training Loss: {}"
        print(msg.format(i, history.history['loss']))
        print("Text generated: " + generate_text("We", 60))

        # Ending time.
        end_time = time.time()
        # Difference between start and end-times.
        time_dif = end_time - start_time
        # Print the time-usage.
        print("Time elapsed: " + str(timedelta(seconds = int(round(time_dif)))))
        print()
    
print("done")

