# silence non silence and save classifier. system input percentage lower and percentage higher threshold

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

os.system('mkdir -p models')

import sys
import config_a

lowerPercent = float(sys.argv[1])
upperPercent = float(sys.argv[2])

scores = config_a.scores
param_grid = config_a.param_grid

FEATUREPATH = config_a.FEATUREPATH

FOLDS_PATH = config_a.FOLDS_PATH

FEAT_DIM = config_a.FEAT_DIM

labelType = config_a.labelType

def getFilesFromPortion(portion):
        t1 = open(portion,'r').read().split('\n')[:-1]
        return [x[:x.find('.')] for x in t1]

foldFileList  = []

for portion in glob.glob(FOLDS_PATH):
        foldFileList.append([x for x in getFilesFromPortion(portion)])

X = np.array([]).reshape(FEAT_DIM,0)
y = np.array([])
lengths = []
for i in xrange(len(foldFileList)):
        length=0
	print 'Reading short fold ' + str(i)
        for f in foldFileList[i]:
                filename = FEATUREPATH+str(i)+'/'+f
                y1 = pickle.load(open(filename+'_y'+labelType,'r'))
                x1 = pickle.load(open(filename+'_X','r'))
                y1 = np.asarray(y1)
                if y1.shape[0] < x1.shape[1]:
                        x1 = x1[:,:y1.shape[0]]
                else:
                        y1 = y1[:x1.shape[1]]
                length+=y1.shape[0]
                X = np.hstack((X,x1))
                y = np.concatenate((y,y1))
        lengths.append(length)

boundaries = [sum(lengths[:x]) for x in xrange(len(lengths)+1)]
idxs = [(boundaries[i],boundaries[i+1]) for i in xrange(len(boundaries)-1)]
X = X.T


# 1 near
# 2 all = near+far



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


def getTotalEnergyVector(X_in):     # given a single list of wav paths, return their aggregate lower-upper% vector
        EnergySt = X_in.T[1, :]
        E = np.sort(EnergySt)
        L1 = int(len(E) * lowerPercent * 0.01)
        L2 = int(len(E) * upperPercent * 0.01)
        T1 = np.mean(E[0:L1]) + 0.000000000000001
        T2 = np.mean(E[-L2:-1]) + 0.000000000000001                # compute "higher" 10% energy threshold
        Class1 = X_in.T[:, np.where(EnergySt <= T1)[0]]         # get all features that correspond to low energy
        # Class1 = ShortTermFeatures[1,:][np.where(EnergySt <= T1)[0]]         # purely energy
        Class2 = X_in.T[:, np.where(EnergySt >= T2)[0]]         # get all features that correspond to high energy
        # Class2 = ShortTermFeatures[1,:][np.where(EnergySt >= T2)[0]]         # purely energy
        featuresSS = [Class1.T, Class2.T]                                  # form the binary classification task
        #[featuresNormSS, MEANSS, STDSS] = aT.normalizeFeatures(featuresSS) # normalize to 0-mean 1-std
        #[X,y] = featureListToVectors(featuresNormSS)
        [X,y] = featureListToVectors(featuresSS)
        return X,y



param_list = list(ParameterGrid(param_grid))
best_f1 = -1
best_f1_i = -1

param_score_dict = {}
for i in scores:
        param_score_dict[i] = {}

print 'Hyperparametrizing short'

for p,params in enumerate(param_list):
	print p,
        params['n_jobs'] = 28
        splitGenerator = yieldIdxs(foldFileList,idxs)
        f1_list = []
        precision_list = []
        recall_list = []
        for i in xrange(len(foldFileList)):
		clf = pickle.load(open('models/a_classifier_template.p','r'))
		clf.set_params(**params)
                clf_test = pickle.load(open('models/a_classifier_template.p','r'))
		clf_test.set_params(**params)
                trainIdx,testIdx = splitGenerator.next()
                X_train = X[trainIdx]
                X_test  = X[testIdx]
                X_train_energy,y_train_energy = getTotalEnergyVector(X_train)
                X_test_energy,y_test_energy = getTotalEnergyVector(X_test)
                fit1 = clf.fit(X_train_energy,y_train_energy)
                fit2 = clf_test.fit(X_test_energy,y_test_energy)
                y_true = clf_test.predict(X_test)
                y_pred = clf.predict(X_test)
                f1_list.append(f1_score(y_true,y_pred,average='weighted'))
                if f1_list[-1] > best_f1:
                        best_f1   = f1_list[-1]
                        best_f1_i = i
                precision_list.append(precision_score(y_true,y_pred,average='weighted'))
                recall_list.append(recall_score(y_true,y_pred,average='weighted'))
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

print 'Given these params, training a model over all.'

all_classifier = pickle.load(open('models/a_classifier_template.p','r'))
all_classifier.set_params(**param_list[best_param_idx])
all_classifier.fit(X,y)

pickle.dump(all_classifier,open('models/a_classifier_all.p','w'))
print 'Model models/a_classifier_all.p saved'


