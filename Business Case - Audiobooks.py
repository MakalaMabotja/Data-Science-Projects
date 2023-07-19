import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing

#first Ineed to preprocess the data for our ML algorith
#the data was cleaned as a csv file on excel however Iwould like to showcase if the data is clean

#raw_data = pd.read_csv('Audiobooks_data.csv')
raw_data = np.loadtxt('Audiobooks_data.csv',delimiter=',')

#print(raw_data.iloc[:,:-1])
#Now I need to load the inputs and targets from our imported array 

inputs_raw = raw_data[:,1:-1].astype(float)
targets_raw = raw_data[:,-1].astype(int)

#balance the data set to ensure I have as many 1s as 0s by undersampling
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

#Our next step is to get the required entries and to preprocess them and then to shuffle them for our model
unscaled_inputs_equal_priors = np.delete(inputs_raw, indices_to_remove, axis=0)
targets_equal_priors = np.delete(targets_raw, indices_to_remove, axis=0)

scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)
shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]

#I need to split our dataset for training, validation & testing using a 80-10-10 split
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

# I need to ensure that our shuffled dataset is still balanced
# the output I expect is for the average value of our targets to 50% (50% of the vlues being 1s)

print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)

#Save into the correct format for tensorflow:
np.savez('Audiobooks_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Audiobooks_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('Audiobooks_data_test', inputs=test_inputs, targets=test_targets)
#this can be achieved with convert_to_tensor method however TF requires npz file types
npz = np.load('Audiobooks_data_train.npz')
npz = np.load('Audiobooks_data_validation.npz')
npz = np.load('Audiobooks_data_test.npz')

train_inputs,train_targets = npz['inputs'], npz['targets']
validation_inputs, validation_targets = npz['inputs'], npz['targets']
test_inputs, test_targets = npz['inputs'], npz['targets']

#I will define our hyperparameters
input_size = 10
output_size=2
hidden_layer_size=50

#The model that I want to build will be sequential
model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_layer_size,activation='relu'),
        tf.keras.layers.Dense(hidden_layer_size,activation='relu'),
        tf.keras.layers.Dense(hidden_layer_size,activation='selu'),
        tf.keras.layers.Dense(hidden_layer_size,activation='selu'),
        tf.keras.layers.Dense(output_size, activation='softmax')
                            ])

#I now want to compile our model by defining the onjective (loss) function and optimization algorithm
model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = 0.005),
              loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#Another hyperparameter I want to define is the batch_size which facilitates the learning process for our SGD
batch_size=500
max_epoch=50

#Next I fit the model to our data for the machine to do the actual learning
model.fit(train_inputs,train_targets,epochs=max_epoch,
          validation_data=(validation_inputs,validation_targets),
          verbose=2)

#Now I want to test my model to ensure that it has learnt and idealy to make sure that no overfitting has occured
# Ideally the train accuracy should be similar if not the same as the validation & testing accuracy          
test_loss, test_accuracy = model.evaluate(test_inputs, test_targets, verbose=0)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')
print(model.summary())
print(model.get_weights())
model.reset_states()
