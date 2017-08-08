# given a silence classifier, use it to generate audacity files and evaluate performance.
# user can adjust classifier as they want
# system input - wavFile long/short outfileFolder


import numpy as np
import os
import glob
import cPickle as pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from pyAudioAnalysis import audioFeatureExtraction as aF
from pyAudioAnalysis import audioTrainTest as aT
from scipy import stats

import labelTools
import extractFeatures

import sys


stWin = extractFeatures.stWin
stStep = extractFeatures.stStep

inputWav = sys.argv[1]

classModel = sys.argv[2]

outfileFolder = sys.argv[3]

speechModel = sys.argv[4]

clf = pickle.load(open(classModel,'r'))

clfSpeech = pickle.load(open('speech_classifier_'+speechMode+'.p','r'))


X_test = extractFeatures.getRawStVectorPerWav(inputWav,stWin,stStep)
X_test = X_test.T
y_class_test = clf.predict(X_test)
y_class_sil = clfSpeech.predict(X_test)
y_out = y_class_test*y_class_sil

flag = 0
for i in xrange(y_out.shape[0]):
	if y_out[i]==0:
		if flag==1:
			end=i
			vote = stats.mode(y_out[start:end])[0][0]
			y_out[start:end] = stats.mode(y_out[start:end])[0][0]
			flag=0
	else:
		if flag==0:
			start=i
			flag=1


# silDict = {'SIL':0,'SPE':1}
classDict        = {'CHI':1,'MOT':2,'FAT':3,'OCH':4,'OAD':5,'OTH':0}
classDictBinary  = {'CHI':1,'MOT':2,'FAT':2,'OCH':1,'OAD':2,'OTH':0} # child adult
classDictTernary = {'CHI':1,'MOT':2,'FAT':3,'SIL':0} #'OCH':1,'OAD':0,'OTH':0} # child male female
medianame = extractFeatures.basename(inputWav)
labelTools.writeToStm(y_out,classDictTernary,medianame,outfileFolder+'/'+medianame+'.stm')
labelTools.writeToAudacity(y_out,classDictTernary,outfileFolder+'/'+medianame+'.txt')
pickle.dump(y_out,open(outfileFolder+'/'+medianame+'_y_out.p','w'))

os.system('cp '+inputWav+' '+outfileFolder+'/'+medianame+'.wav')



