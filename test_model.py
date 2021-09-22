import numpy as np
import glob
import os
import keras
from keras.models import Model
from keras.layers import Input, Dense, GRU, CuDNNGRU, CuDNNLSTM
from keras import optimizers
import h5py
from sklearn.model_selection import train_test_split
from keras.models import load_model
import sklearn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

def language_name(index):
	if index == 0:
		return "english"
	elif index == 1:
		return "hindi"
	elif index==2:
		return "tamil"
	else:
		return "telugu"



codePath = './test/'
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
Y_test = np.vstack((english_labels, hindi_labels,tamil_labels,telugu_labels))


X_test = np.vstack((list_english_mfcc,list_hindi_mfcc,list_tamil_mfcc,list_telugu_mfcc))

print(X_test.shape)
del list_english_mfcc
del list_hindi_mfcc
del list_tamil_mfcc
del list_telugu_mfcc



print("loading model..")
streaming_input = Input(name='streaming_input', batch_shape=(4,sequence_length, 39))
pred_layer1 = CuDNNLSTM(39, return_sequences=True, name='layer1', stateful=True)(streaming_input)
pred_layer2 = CuDNNLSTM(13, return_sequences=True, name='layer2')(pred_layer1)
pred_layer3 = Dense(80, activation='tanh', name='layer3')(pred_layer2)
pred_output = Dense(4, activation='softmax', name='rnn_output')(pred_layer3)
streaming_model = Model(inputs=streaming_input, outputs=pred_output)
streaming_model.load_weights('sld.hdf5')
optimizer = optimizers.Adam(decay=1e-5)
streaming_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
print('model loaded successfully.')
accuracy=streaming_model.evaluate(X_test,Y_test,batch_size=4)
print(accuracy)
y_pred=streaming_model.predict(X_test)
y_pred=np.argmax(y_pred,axis=-1)
print(y_pred.shape)
Y_test=np.argmax(Y_test,axis=-1)
'''
Y_test_new=[]
y_pred_new=[]
for i in range(4000):
	Y_test_new.append(keras.utils.to_categorical(Y_test[i,:],num_classes=4))
	y_pred_new.append(keras.utils.to_categorical(y_pred[i,:],num_classes=4))
'''


cm=confusion_matrix(y_pred.ravel(),Y_test.ravel())


plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")

plt.title('Confusion matrix ')
plt.colorbar()
plt.show()
