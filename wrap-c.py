# Given silence/nonsilence classifier, use it to learn within speech
# for long, it already has segments so it is fine. this is for short.

import numpy as np
import os
import glob
import cPickle as pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import sys

try:
	speechModel = sys.argv[1]
except:
	speechModel = 'models/a_classifier_all.p'


try:
	labelType = sys.argv[2]
except:
	labelType = 'class'

try:
	outputFile   = sys.argv[3]
except:
	outputFile   = 'models/c_classifier_all.p'


scores = config_c.scores
param_grid = config_c.param_grid

FEATUREPATH = config_c.FEATUREPATH

FOLDS_PATH = config_c.FOLDS_PATH

FEAT_DIM = config_c.FEAT_DIM

labelType = config_c.labelType


def getFilesFromPortion(portion):
        t1 = open(portion,'r').read().split('\n')[:-1]
        return [x[:x.find('.')] for x in t1]

foldFileList  = []

for portion in glob.glob(FOLDS_PATH):
        foldFileList.append([x for x in getFilesFromPortion(portion)])


folders = glob.glob(FEATUREPATH+'*')

clfSpeech = pickle.load(open(speechModel,'r'))



X = np.array([]).reshape(FEAT_DIM,0)
y = np.array([])
lengths = []
for i in xrange(len(foldFileList)):
	length=0
	for f in foldFileList[i]:
		print 'Reading '+str(f)
		filename = FEATUREPATH+str(i)+'/'+f
		y1 = pickle.load(open(filename+'_y'+labelType,'r'))
		x1 = pickle.load(open(filename+'_X','r'))
		y1 = np.asarray(y1)
		if y1.shape[0] < x1.shape[1]:
			x1 = x1[:,:y1.shape[0]]
		else:
			y1 = y1[:x1.shape[1]]
		y_speech = clfSpeech.predict(x1.T)
		y1 = y1[y_speech>0]
		x1 = x1[:,y_speech>0]
		x1 = x1[:,y1>0]
		y1 = y1[y1>0]
		length+=y1.shape[0]
		X = np.hstack((X,x1))
		y = np.concatenate((y,y1))
	lengths.append(length)

boundaries = [sum(lengths[:x]) for x in xrange(len(lengths)+1)]
idxs = [(boundaries[i],boundaries[i+1]) for i in xrange(len(boundaries)-1)]
X = X.T



def yieldIdxs(foldFileList,idxs):
        for i in xrange(len(foldFileList)):
                trainidxs = []
                for j in xrange(len(foldFileList)):
                        if i!=j:
                                trainidxs += range(idxs[j][0],idxs[j][1])
                testidxs = range(idxs[i][0],idxs[i][1])
                yield trainidxs,testidxs


def featureListToVectors(featureList):
        X = np.array([])
        Y = np.array([])
        for i, f in enumerate(featureList):
                if i == 0:
                        X = f
                        Y = i * np.ones((len(f), 1))
                else:
                        X = np.vstack((X, f))
                        Y = np.append(Y, i * np.ones((len(f), 1)))
        return (X, Y)





param_list = list(ParameterGrid(param_grid))

best_f1 = -1
best_f1_i = -1

param_score_dict = {}
for i in scores:
        param_score_dict[i] = {}

print 'Hyperparametrizing for class'

for p,params in enumerate(param_list):
        print p,
        params['n_jobs'] = 28
        splitGenerator = yieldIdxs(foldFileList,idxs)
        f1_list = []
        precision_list = []
        recall_list = []
        for i in xrange(len(foldFileList)):
		clf = pickle.load(open('models/c_classifier_template.p','r'))
                clf.set_params(**params)
		clf_test = pickle.load(open('models/c_classifier_template.p','r'))
                clf_test.set_params(**params)
                trainIdx,testIdx = splitGenerator.next()
                X_train = X[trainIdx]
                X_test  = X[testIdx]
		y_train = y[trainIdx]
		y_test  = y[testIdx]
		clf.fit(X_train,y_train)
                y_pred = clf.predict(X_test)
                f1_list.append(f1_score(y_test,y_pred,average='weighted'))
                if f1_list[-1] > best_f1:
                        best_f1   = f1_list[-1]
                        best_f1_i = i
                precision_list.append(precision_score(y_test,y_pred,average='weighted'))
                recall_list.append(recall_score(y_test,y_pred,average='weighted'))
        param_score_dict['f1'][p]       =       np.mean(f1_list)
        param_score_dict['precision'][p]=       np.mean(precision_list)
        param_score_dict['recall'][p]   =       np.mean(recall_list)
        print param_score_dict['f1'][p]

tempMax = -1
best_param_idx = None
for k,v in param_score_dict['f1'].items():
        if v>tempMax:
                tempMax=v
                best_param_idx = k

print 'Best params over F1 are: ' + str(param_list[best_param_idx])
print 'Corresponding F1       : ' + str(param_score_dict['f1'][best_param_idx])
print 'Corresponding precision: ' + str(param_score_dict['precision'][best_param_idx])
print 'Corresponding recall   : ' + str(param_score_dict['recall'][best_param_idx])


print 'Training global model with these parameters.'
all_classifier_class = pickle.load(open('models/c_classifier_template.p','r'))
all_classifier_class.set_params(**param_list[best_param_idx])
all_classifier_class.fit(X,y)

pickle.dump(class_rfc,open(outputFile,'w'))

print 'Model '+outputFile+' saved.'


























