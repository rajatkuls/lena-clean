import glob
import os

f = glob.glob('../data/VanDam/*wav')

t1 = [x[x.rfind('/')+1:] for x in f]
t2 = [x[:x.rfind('.')] for x in t1]


for m in t2:
	print m
	os.system('python wrap-d.py ../data/VanDam/'+m+'.wav class_short_short_class2.p hyp/test short')
	os.system('python wrap-e.py ../data/VanDam/'+m+'.stm hyp/test/'+m+'_y_out.p')





