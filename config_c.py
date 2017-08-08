import cPickle as pickle

# Portion a is to access the vectors generated in preprocessing to train a silence/noise classifier

# The VanDam daylong recordings contain near and far annotations. 
# I use _1 modifier to signify near only,
# 	_2 modifier to signify near+far both.					


# These scores are calculated during the hyperparametrization
scores = ['f1','precision','recall']

# Define the classifier here. Example Random Forest is used.
# This will be initialized when this config module is imported
# remember to import the classifier in wrap-c also
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
pickle.dump(rfc,open('models/a_classifier_template.p','w'))
# Use corresponding parameter search space
param_grid = {'n_estimators': [1,10], 'max_features': ['auto', 'sqrt']}






FEATUREPATH = '../data/features/st/VanDam/'

FOLDS_PATH = '../data/folds/VanDam/portion*'

FEAT_DIM = 34

# The labelType is to differentiate between the silence/noise and classification data
labelType = 'sil'









## Extra

WAV_PATH = '../data/VanDam/'
DATA_PATH = '../data/'






