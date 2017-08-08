# This is a config file to store the various mappings - Annotator to standard label, and label to integer

import glob
import os

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


classDict        = {'CHI':1,'MOT':2,'FAT':3,'OCH':4,'OAD':5,'OTH':0}
classDictBinary  = {'CHI':1,'MOT':2,'FAT':2,'OCH':1,'OAD':2,'OTH':0} # child adult
classDictTernary = {'CHI':1,'MOT':2,'FAT':3,'OCH':1,'OAD':0,'OTH':0} # child male female

	








