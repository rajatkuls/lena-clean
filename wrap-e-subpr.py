import numpy as np
import os
import glob
import cPickle as pickle
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import extractFeatures
import sys

true_stm = sys.argv[1]
output_vec = sys.argv[2]

y_true = extractFeatures.getLabelPerWav(true_stm,extractFeatures.stStep,extractFeatures.classMapTernaryFn,extractFeatures.shortSpeechDict)

y_pred = pickle.load(open(output_vec,'r'))

if len(y_pred)<len(y_true):
	y_true = y_true[:len(y_pred)]

if len(y_true)<len(y_pred):
        y_pred = y_pred[:len(y_true)]

x = true_stm[true_stm.rfind('/')+1:]
outfile = x[:x.rfind('.')]

f = open('hyp/avg/'+outfile,'w')

f.write('F1\n')
f.write(str(f1_score(y_true,y_pred,average='weighted')))
f.write('\n')
f.write('Precision\n')
f.write(str(precision_score(y_true,y_pred,average='weighted')))
f.write('\n')
f.write('Recall\n')
f.write(str(recall_score(y_true,y_pred,average='weighted')))
f.write('\n')





