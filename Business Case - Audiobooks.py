import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing

#first we need to preprocess the data for our ML algorith
#the data was cleaned as a csv file on excel however we would like to showcase if the data is clean

#raw_data = pd.read_csv('Audiobooks_data.csv')
raw_data = np.loadtxt('Audiobooks_data.csv',delimiter=',')

#print(raw_data.iloc[:,:-1])
#Now we need to load the inputs and targets from our dataframe - might need to convert
#to numpy array if we frin challanges later

inputs_raw = raw_data[:,1:-1].astype(float)
targets_raw = raw_data[:,-1].astype(int)

#balance the data set to ensure we have as many 1s as 0s by undersampling
num_one_targets = int(np.sum(targets_raw))
zero_targets_counter = 0
indices_to_remove=[]

# Count the number of targets that are 0. 
# Once there are as many 0s as 1s, mark entries where the target is 0.
for i in range(targets_raw.shape[0]):
    if targets_raw[i] == 0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

unscaled_inputs_equal_priors = np.delete(inputs_raw, indices_to_remove, axis=0)
targets_equal_priors = np.delete(targets_raw, indices_to_remove, axis=0)

scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)
shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]

#We need to split our dataset for training, validation & testing
samples_count = shuffled_inputs.shape[0]
train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count

train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]
validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

# WE need to ensure that our shuffled dataset is still balanced
# the output we expect is for the average value of our targets to 50% (50% of the vlues being 1s)

print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)

#Save into the correct format for tensorflow:
np.savez('Audiobooks_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Audiobooks_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('Audiobooks_data_test', inputs=test_inputs, targets=test_targets)
#this can be achieved with convert_to_tensor which we'll experiment with later
npz = np.load('Audiobooks_data_train.npz')
npz = np.load('Audiobooks_data_validation.npz')
npz = np.load('Audiobooks_data_test.npz')
#see above and see if we can achieve this coming step in one action without the loading & saving

train_inputs,train_targets = npz['inputs'], npz['targets']
validation_inputs, validation_targets = npz['inputs'], npz['targets']
test_inputs, test_targets = npz['inputs'], npz['targets']

input_size = 10
output_size=2
hidden_layer_size=200

model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_layer_size,activation='relu'),
        tf.keras.layers.Dense(hidden_layer_size,activation='relu'),
        tf.keras.layers.Dense(hidden_layer_size,activation='selu'),
        tf.keras.layers.Dense(hidden_layer_size,activation='selu'),
        tf.keras.layers.Dense(output_size, activation='softmax')
                            ])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

batch_size=1000
max_epoch=25

model.fit(train_inputs,train_targets,epochs=max_epoch,
          validation_data=(validation_inputs,validation_targets),
          verbose=2)

print(model.summary())
model.reset_states()