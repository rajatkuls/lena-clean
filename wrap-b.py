# given a silence classifier, use it to generate audacity files and evaluate performance.
# user can adjust thresholds of wrap-a using this as diagnostic
# system input - wavFile pathToModel outputFolder


import labelTools
import extractFeatures
import sys
import os
import numpy as np
import cPickle as pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import config_b
# Ensure that your classifier of choice is imported here

silDict = labelTools.silDict

stWin = extractFeatures.stWin
stStep = extractFeatures.stStep

inputWav = sys.argv[1]

try:
	model = sys.argv[2] # model saved by wrap-a 
	outfileFolder = sys.argv[3]
except:
	model = 'models/a_classifier_all.p'
	outfileFolder = 'models'

clf = pickle.load(open(model,'r'))

X_test = extractFeatures.getRawStVectorPerWav(inputWav,stWin,stStep)
X_test = X_test.T
y_sil_test = clf.predict(X_test)

medianame = extractFeatures.basename(inputWav)
labelTools.writeToStm(y_sil_test,silDict,medianame,outfileFolder+'/'+medianame+'.stm')
labelTools.writeToAudacity(y_sil_test,silDict,outfileFolder+'/'+medianame+'.txt')

os.system('cp '+inputWav+' '+outfileFolder+'/'+medianame+'.wav')


