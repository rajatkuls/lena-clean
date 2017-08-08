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

mode = sys.argv[1]
speechMode = sys.argv[2]


try:
	labelType = sys.argv[3]
except:
	labelType = 'class'


ST_FEATUREPATH = '../data/features/st/VanDam/'
ST_FEATUREPATH1 = '../data/features/st/VanDam1/'

FOLDS_PATH = '../data/folds/VanDam/portion*'
FOLDS_PATH1 = '../data/folds/VanDam1/portion*'


WAV_PATH = '../data/VanDam/'
WAV1_PATH = '../data/VanDam1/'  # I can't write wavs to original location
DATA_PATH = '../data/'

FEAT_DIM = 34




def getFilesFromPortion(portion):
        t1 = open(portion,'r').read().split('\n')[:-1]
        return [x[:x.find('.')] for x in t1]

foldFileList  = []
foldFileList1 = []

for portion in glob.glob(FOLDS_PATH):
        foldFileList.append([x for x in getFilesFromPortion(portion)])

for portion in glob.glob(FOLDS_PATH1):
        foldFileList1.append([x for x in getFilesFromPortion(portion)])


folders = glob.glob(ST_FEATUREPATH+'*')

clfSpeech = pickle.load(open('speech_classifier_'+speechMode+'.p','r'))



if mode=='short':
X = np.array([]).reshape(FEAT_DIM,0)
y = np.array([])
lengths = []
for i in xrange(len(foldFileList)):
	length=0
	for f in foldFileList[i]:
		print 'Reading '+str(f)
		filename = ST_FEATUREPATH+str(i)+'/'+f
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
	print X.shape

else:
	X = np.array([]).reshape(FEAT_DIM,0)
	y = np.array([])
	# y2 = np.array([])
	lengths = []
	for i in xrange(len(foldFileList1)):
		length=0
		for f in foldFileList1[i]:
			print 'Reading '+str(f)
			filename = ST_FEATUREPATH1+str(i)+'/'+f
			y1_temp = pickle.load(open(filename+'_y'+'sil_near','r'))
			# y2_temp = pickle.load(open(filename+'_y'+'sil_all','r'))
			x1 = pickle.load(open(filename+'_X','r'))
			y1_temp = np.asarray(y1_temp)
			# y2_temp = np.asarray(y2_temp)
			# assert y1_temp.shape[0]==y2_temp.shape[0]
			if y1_temp.shape[0] < x1.shape[1]:
				x1 = x1[:,:y1_temp.shape[0]]
			else:
				y1_temp = y1_temp[:x1.shape[1]]
				# y2_temp = y2_temp[:x1.shape[1]]
			y_speech = clfSpeech.predict(x1.T)
			y1_temp = y1_temp[y_speech>0]
			x1 = x1[:,y_speech>0]
			x1 = x1[:,y1_temp>0]
                        y1_temp = y1_temp[y1_temp>0]
			length+=y1_temp.shape[0]
		assert y1_temp.shape[0]==x1.shape[1]
		X = np.hstack((X,x1))
		y = np.concatenate((y,y1_temp))
		# y2 = np.concatenate((y2,y2_temp))
		lengths.append(length)

	foldFileList=foldFileList1
	boundaries = [sum(lengths[:x]) for x in xrange(len(lengths)+1)]
	idxs = [(boundaries[i],boundaries[i+1]) for i in xrange(len(boundaries)-1)]
	X = X.T
	print X.shape



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




scores = ['f1','precision','recall']
param_grid = {'n_estimators': [10], 'max_features': ['auto']}
param_grid = {'n_estimators': [1,10], 'max_features': ['auto', 'sqrt', 'log2']}

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
                rfc = RandomForestClassifier(**params)
                rfc_test = RandomForestClassifier(**params)
                trainIdx,testIdx = splitGenerator.next()
                X_train = X[trainIdx]
                X_test  = X[testIdx]
		y_train = y[trainIdx]
		y_test  = y[testIdx]
		rfc.fit(X_train,y_train)
                y_pred = rfc.predict(X_test)
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
class_rfc = RandomForestClassifier(**param_list[best_param_idx])
class_rfc.fit(X,y)

pickle.dump(class_rfc,open('class_'+mode+'_'+speechMode+'.p','w'))

print 'Model class_'+mode+'_'+speechMode+'_'+labelType+'.p saved.'


























