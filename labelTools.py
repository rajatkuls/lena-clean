# This is a config file to store the various mappings - Annotator to standard label, and label to integer

import glob
import os
from decimal import Decimal, getcontext
import extractFeatures_config
import numpy as np


WAV_PATH = extractFeatures_config.WAV_PATH
DATA_PATH = extractFeatures_config.DATA_PATH
OUTPUT_FEATURES_PATH = extractFeatures_config.OUTPUT_FEATURES_PATH

FRAME_WIDTH = extractFeatures_config.FRAME_WIDTH
INV_FRAME_WIDTH = extractFeatures_config.INV_FRAME_WIDTH
MIN_DUR = extractFeatures_config.MIN_DUR

label_map = {
    "ADU": "OAD", # adult
    "ALL": "OCH", # all_together child
    "ATT": "OAD", # toll booth attendant
    "AU2": "OAD", # audience2 adult
    "AU3": "OAD", # audience3 adult
    "AUD": "OAD", # audience1 adult
    "AUN": "OAD", # aunt
    "BAB": "OCH", # sibling
    "BEN": "OCH", # Ben child
    "BRO": "OCH", # brother
    "CHI": "CHI", # target child
    "ELE": "OAD", # unidentified adult
    "EVA": "MOT", # Eva mother
    "FAT": "FAT", # father
    "GMA": "OAD", # grandmother
    "GPA": "OAD", # grandfather
    "GRA": "OAD", # grandmother
    "GRF": "OAD", # grandfather
    "GRM": "OAD", # grandmother
    "JAN": "OAD", # Janos visitor
    "JIM": "OAD", # Jim grandfather
    "JOE": "OCH", # Joey child
    "KID": "OCH", # child
    "KUR": "OAD", # Kurt adult
    "LOI": "OAD", # Louise aunt
    "MAD": "OCH", # Madeleine child
    "MAG": "CHI", # Maggie target child
    "MAN": "OAD", # man
    "MAR": "OCH", # Mark brother
    "MOT": "MOT", # mother
    "NEI": "OAD", # neighbor adult
    "OTH": "OAD", # other adult
    "PAR": "OAD", # participant adult
    "PAR1": "OAD", # participant adult
    "PAR2": "OAD", # participant adult/child *
    "PAR3": "OAD", # participant adult
    "PAR4": "OAD", # participant adult
    "PER": "OAD", # person
    "ROS": "CHI", # Ross target child
    "SI1": "OCH", # sibling
    "SI2": "OCH", # sibling
    "SI3": "OCH", # sibling
    "SIB": "OCH", # sibling
    "SP01": "OAD", # adult
    "SPO1": "OAD", # female adult
    "TEA": "OAD", # adult
    "TEL": "OTH", # media
    "TOY": "OTH", # toy
    "UN1": "OCH", # adult/child *
    "UN2": "OCH", # child
    "UN3": "OAD", # adult/child *
    "UN4": "OCH", # child
    "UNK": "OAD", # uncle
    "VAC": "OAD", # unidentified person
    "VIS": "OAD", # visitor
    "WOM": "OAD", # woman
}


label_map1_near = {
	"CHN": "CHI", # child near clear
	"FAN": "MOT", # female adult near
	"MAN": "FAT", # male adult near
	"CXN": "OCH", # other child near
}

label_map1_far = {
	"CHF": "CHI", # child far
	"FAF": "MOT", # mother far
	"MAF": "FAT", # father far
	"CXF": "OCH", # other child far
}

silDict		 = {'SIL':0,'SPE':1}
classDict        = {'CHI':1,'MOT':2,'FAT':3,'OCH':4,'OAD':5,'OTH':6} # child mother father otherchild otheradult
classDictBinary  = {'CHI':1,'MOT':2,'FAT':2,'OCH':1,'OAD':2,'OTH':3} # child adult
classDictTernary = {'CHI':1,'MOT':2,'FAT':3,'OCH':1,'OAD':4,'OTH':4} # child male female




def convertToFrames(stmFile):
        stm = open(stmFile,'r').read().split('\n')[:-1]
        seg = []
        for line in stm:
                start = float(line.split(' ')[3])
                end = float(line.split(' ')[4])
                t1 = [(start+i*FRAME_WIDTH,start+(i+1)*FRAME_WIDTH,line.split(' ')[1]) for i in xrange(int((end-start+MIN_DUR)*INV_FRAME_WIDTH))]
                if str(start)!=str(t1[0][0]) or str(end)!=str(t1[-1][1]):
                        print start,end
                        print t1[0][0],t1[-1][1]
                        print
                seg.extend(t1)
        return seg


stStep = extractFeatures_config.stStep

def stmNewLine(medianame,label,start,end):
        return '\t'.join([medianame,label,medianame+'_'+label,str(start),str(end)]) + '\n'

def writeToStm(y,labelsDict,medianame,outfilename):
        revDict = {v:k for k,v in labelsDict.items()}
	revDict[silDict['SIL']] = 'SIL'
        y = np.asarray(y)
        boundaries = list((y[:-1] != y[1:]).nonzero()[0] + 1) + [y.shape[0]-1]
        labels = [revDict[y[x-1]] for x in boundaries]
        curFrames=0
        assert len(boundaries)==len(labels)
        stm = ''
        for i in xrange(len(labels)):
                stm+=stmNewLine(medianame,labels[i],curFrames*0.05,boundaries[i]*0.05)
                curFrames = boundaries[i]
        f = open(outfilename,'w')
        f.write(stm)
        f.close()

def audacityNewLine(start,end,label):
        return '\t'.join([str(start),str(end),label]) + '\n'


def writeToAudacity(y,labelsDict,outfilename):
        revDict = {v:k for k,v in labelsDict.items()}
	revDict[silDict['SIL']] = 'SIL'
        y = np.asarray(y)
        boundaries = list((y[:-1] != y[1:]).nonzero()[0] + 1) + [y.shape[0]-1]
        labels = [revDict[y[x-1]] for x in boundaries]
        curFrames=0
        assert len(boundaries)==len(labels)
        txt = ''
        for i in xrange(len(labels)):
                txt+=audacityNewLine(curFrames*0.05,boundaries[i]*0.05,labels[i])
                curFrames = boundaries[i]
        f = open(outfilename,'w')
        f.write(txt)
        f.close()





