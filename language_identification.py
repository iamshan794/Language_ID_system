import numpy as np
import glob
import os
import keras
from keras.models import Model
from keras.layers import Input, Dense, GRU, CuDNNGRU, CuDNNLSTM,Dropout
from keras import optimizers
import h5py
from sklearn.model_selection import train_test_split
from keras.models import load_model


def language_name(index):
	if index == 0:
		return "english"
	elif index == 1:
		return "hindi"
	elif index==2:
		return "tamil"
	else:
		return "telugu"
    

# ---------------------------BLOCK 1------------------------------------
# COMMENT/UNCOMMENT BELOW CODE BLOCK -
# Below code extracts mfcc features from the files provided into a dataset
codePath = './train/'
num_mfcc_features = 39

english_mfcc = np.array([]).reshape(0, num_mfcc_features)
for file in glob.glob(codePath + 'english/*.npy'):
    current_data = np.load(file)
    english_mfcc = np.vstack((english_mfcc, current_data))
print("English done")
hindi_mfcc = np.array([]).reshape(0, num_mfcc_features)
for file in glob.glob(codePath + 'hindi/*.npy'):
    current_data = np.load(file)
    hindi_mfcc = np.vstack((hindi_mfcc, current_data))
print("Hindi done")
tamil_mfcc = np.array([]).reshape(0, num_mfcc_features)
for file in glob.glob(codePath + 'tamil/*.npy'):
    current_data = np.load(file)
    tamil_mfcc = np.vstack((tamil_mfcc, current_data))
print("tamil done")
telugu_mfcc = np.array([]).reshape(0, num_mfcc_features)
for file in glob.glob(codePath + 'telugu/*.npy'):
    current_data = np.load(file)
    telugu_mfcc = np.vstack((telugu_mfcc, current_data))
print("telugu done")

# Sequence length is 10 seconds
sequence_length = 1000
list_english_mfcc = []
num_english_sequence = int(np.floor(len(english_mfcc)/sequence_length))
for i in range(num_english_sequence):
    list_english_mfcc.append(english_mfcc[sequence_length*i:sequence_length*(i+1)])
list_english_mfcc = np.array(list_english_mfcc)
english_labels = np.full((num_english_sequence, 1000, 4), np.array([1, 0,0,0]))

list_hindi_mfcc = []
num_hindi_sequence = int(np.floor(len(hindi_mfcc)/sequence_length))
for i in range(num_hindi_sequence):
    list_hindi_mfcc.append(hindi_mfcc[sequence_length*i:sequence_length*(i+1)])
list_hindi_mfcc = np.array(list_hindi_mfcc)
hindi_labels = np.full((num_hindi_sequence, 1000, 4), np.array([0, 1,0,0]))


list_tamil_mfcc = []
num_tamil_sequence = int(np.floor(len(tamil_mfcc)/sequence_length))
for i in range(num_tamil_sequence):
    list_tamil_mfcc.append(tamil_mfcc[sequence_length*i:sequence_length*(i+1)])
list_tamil_mfcc = np.array(list_tamil_mfcc)
tamil_labels = np.full((num_tamil_sequence, 1000, 4), np.array([0,0, 1,0]))

list_telugu_mfcc = []
num_telugu_sequence = int(np.floor(len(telugu_mfcc)/sequence_length))
for i in range(num_telugu_sequence):
    list_telugu_mfcc.append(telugu_mfcc[sequence_length*i:sequence_length*(i+1)])
list_telugu_mfcc = np.array(list_telugu_mfcc)
telugu_labels = np.full((num_telugu_sequence, 1000, 4), np.array([0,0,0,1]))



del english_mfcc
del hindi_mfcc
del tamil_mfcc
del telugu_mfcc

total_sequence_length = num_english_sequence + num_hindi_sequence +num_telugu_sequence+num_tamil_sequence
Y_train = np.vstack((english_labels, hindi_labels,tamil_labels,telugu_labels))


X_train = np.vstack((list_english_mfcc,list_hindi_mfcc,list_tamil_mfcc,list_telugu_mfcc))


del list_english_mfcc
del list_hindi_mfcc
del list_tamil_mfcc
del list_telugu_mfcc


X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.01)


print("Dataset collected")


class myCallback(keras.callbacks.Callback): 
	def on_epoch_end(self, epoch, logs={}): 
		if(logs.get('val_acc') > 0.94):   
			print("Done")   
			self.model.stop_training = True
thecallback=myCallback()
# ---------------------------BLOCK 3------------------------------------
# Setting up the model for training
DROPOUT = 0.4
RECURRENT_DROP_OUT = 0.2
optimizer = optimizers.Adam(decay=1e-5)
main_input = Input(shape=(sequence_length, 39), name='main_input')

# ### main_input = Input(shape=(None, 64), name='main_input')
# ### pred_gru = GRU(4, return_sequences=True, name='pred_gru')(main_input)
# ### rnn_output = Dense(3, activation='softmax', name='rnn_output')(pred_gru)

layer1 = CuDNNLSTM(39, return_sequences=True, name='layer1')(main_input)
layer2 = CuDNNLSTM(13, return_sequences=True, name='layer2')(layer1)
layer3 = Dense(80,activation='tanh', name='layer3')(layer2)

rnn_output = Dense(4, activation='softmax', name='rnn_output')(layer3)

model = Model(inputs=main_input, outputs=rnn_output)
print('\nCompiling model...')
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
model.summary()
keras.utils.plot_model(model,'lang_id_model.png')
history = model.fit(X_train, Y_train, batch_size=64, epochs=85, validation_data=(X_val, Y_val), shuffle=True, verbose=1)
model.save('sld.hdf5')
# ---------------------------------------------------------------
'''
# --------------------------BLOCK 4-------------------------------------
# Inference Mode Setup
streaming_input = Input(name='streaming_input', batch_shape=(1, 1, 64))
pred_layer1 = CuDNNLSTM(64, return_sequences=True, name='layer1', stateful=True)(streaming_input)
pred_layer2 = CuDNNLSTM(32, return_sequences=True, name='layer2')(pred_layer1)
pred_layer3 = Dense(100, activation='tanh', name='layer3')(pred_layer2)
pred_output = Dense(3, activation='softmax', name='rnn_output')(pred_layer3)
streaming_model = Model(inputs=streaming_input, outputs=pred_output)
streaming_model.load_weights('sld.hdf5')
# streaming_model.summary()
# ---------------------------------------------------------------

# ---------------------------BLOCK 5------------------------------------
# Language Prediction for a random sequence from the validation data set
random_val_sample = np.random.randint(0, X_val.shape[0])
random_sequence_num = np.random.randint(0, len(X_val[random_val_sample]))
test_single = X_val[random_val_sample][random_sequence_num].reshape(1, 1, 64)
val_label = Y_val[random_val_sample][random_sequence_num]
true_label = language_name(np.argmax(val_label))
print("***********************")
print("True label is ", true_label)
single_test_pred_prob = streaming_model.predict(test_single)
pred_label = language_name(np.argmax(single_test_pred_prob))
print("Predicted label is ", pred_label)
print("***********************")
# ---------------------------------------------------------------

# ---------------------------BLOCK 6------------------------------------
## COMMENT/UNCOMMENT BELOW
# Prediction for all sequences in the validation set - Takes very long to run
print("Predicting labels for all sequences - (Will take a lot of time)")
list_pred_labels = []
for i in range(X_val.shape[0]):
    for j in range(X_val.shape[1]):
        test = X_val[i][j].reshape(1, 1, 64)
        seq_predictions_prob = streaming_model.predict(test)
        predicted_language_index = np.argmax(seq_predictions_prob)
        list_pred_labels.append(predicted_language_index)
pred_english = list_pred_labels.count(0)
pred_russian = list_pred_labels.count(1)
pred_mandarin = list_pred_labels.count(2)
print("Number of English labels = ", pred_english)
print("Number of russian labels = ", pred_russian)
print("Number of Mandarin labels = ", pred_mandarin)
# ---------------------------------------------------------------
'''