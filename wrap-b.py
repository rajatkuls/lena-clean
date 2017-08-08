# given a silence classifier, use it to generate audacity files and evaluate performance.
# user can adjust thresholds as they want
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

import finalStmAud
import extractFeatures

import sys


stWin = extractFeatures.stWin
stStep = extractFeatures.stStep

inputWav = sys.argv[1]


mode = sys.argv[2]  # long or short

outfileFolder = sys.argv[3]

clf = pickle.load(open('speech_classifier_'+mode+'.p','r'))

X_test = extractFeatures.getRawStVectorPerWav(inputWav,stWin,stStep)
X_test = X_test.T
y_sil_test = clf.predict(X_test)

silDict = {'SIL':0,'SPE':1}
medianame = extractFeatures.basename(inputWav)
finalStmAud.writeToStm(y_sil_test,silDict,medianame,outfileFolder+'/'+medianame+'.stm')
finalStmAud.writeToAudacity(y_sil_test,silDict,outfileFolder+'/'+medianame+'.txt')

os.system('cp '+inputWav+' '+outfileFolder+'/'+medianame+'.wav')



