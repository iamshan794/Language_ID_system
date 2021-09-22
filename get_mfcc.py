import os
import librosa
import numpy
from numpy import save
languages=os.listdir()[1:]
languages=[languages[i] for i in range(len(languages)) if len(languages[i].split('.'))==1]
print('Languages found'+str(languages))
for alanguage in languages:
	os.chdir(alanguage)
	audio_files=os.listdir()
	for anaudio in audio_files:
		data, rate = librosa.load(anaudio)
		ns=int(data.shape[0]/rate)
		feature39=numpy.array([]).reshape(0,39)
		mfcc_fet=librosa.feature.mfcc(y=data, sr=rate, n_fft=1024, hop_length=512, n_mfcc=13)
		
		mfcc_delta1=librosa.feature.delta(mfcc_fet,order=1)
	
		mfcc_delta2=librosa.feature.delta(mfcc_fet,order=2)
		feature39=numpy.append(mfcc_fet.T,mfcc_delta1.T,axis=1)
		feature39=numpy.append(feature39,mfcc_delta2.T,axis=1)
		print(feature39.shape)
		sname=anaudio.split('.')[0]+'_mfcc_'+'.npy'
		save(sname,feature39)
		print("In "+alanguage+" Converted "+anaudio+" to MFCC")
		
	os.chdir('..')
