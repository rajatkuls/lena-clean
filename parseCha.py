import glob
import pdb
# This contains reading from .cha file and writing to stm

STM_PATH = '../data/VanDam/'

def getName(chaFile):
	return STM_PATH + chaFile[chaFile.rfind('/')+1:chaFile.rfind('.')]  + '.stm'

def stmNewLine(medianame,code,time):
        s = ' '.join([medianame,code,medianame+'_'+code]+[x[:-3]+'.'+x[-3:] for x in time.split('_')])
        # TODO Add more info to s, take more inputs to this function
        return s + '\n'

def stmNewMultiline(medianame,code,line):
	stm = ''
	t1 = line.split('\x15')
	for i in xrange(len(t1)):
		if i%2==1:
			stm += stmNewLine(medianame,code,t1[i])
	return stm

def makeFixedStm(chaFile):
	f = open(chaFile,'r').read().split('@Media')
	header = f[0].split('@')[1:]
	# Consume header
	if 'Font' in header[0]:
		header = header[1:]
	if 'UTF8' not in header[0]:
		raise Exception("Format error")
	header = header[1:]
	if 'Begin' not in header[0]:
		raise Exception("Format error")
	header = header[1:]
	if 'Languages' not in header[0]:
		raise Exception("Format error")
	header = header[1:]
	if 'Participants' not in header[0]:
		raise Exception("Format error")
	participants = header[0]
	header = header[1:]
	# ID Format: @ID: language|corpus|code|age|sex|group|SES|role|education|custom|
	# 			0   1	  2	3   4	5    6	  7	8	9
	IDs = {}
	while len(header)>0:
		info = header[0].split('|')
		IDs[info[2]] = info[7]	# here the mapping is from say MAN code to Male_Adult_Near custom
		header = header[1:]
	for k in IDs.keys():
		if k not in IDs:
			raise Exception("IDs and participants do not match")
	medianame = chaFile[:chaFile.rfind('.')] 
	medianame = medianame[medianame.rfind('/')+1:]
	stm = ''
	t1 = f[1].split('\n*')
	if '\x15' in t1[0]:
		print i
		raise Exception("There's a time marker in the opening of this segment, confirm splitting")
	for line in t1[1:]:
		if line[3]!=':' or not line[:3].isupper():
			print "Line does not start with a 3 digit code like TVN:" + line
		code = line[:line.find(':')]
		if line.count('\x15')==2:
			time = line[line.find('\x15')+1:line.find('\x15',line.find('\x15')+1)]
			stm+= stmNewLine(medianame,code,time)
		elif line.count('\x15') > 2:
			stm+= stmNewMultiline(medianame,code,line)
		else:
			print line
			print chaFile + ' has missing timestamp here\n'
		#check for gaps here:
		try:
			t1 = stm.split('\n')[-3:-1]
			if t1[0].split(' ')[-1] != t1[1].split(' ')[-2]:
				if float(t1[0].split(' ')[-1]) < float(t1[1].split(' ')[-2]):
					print 'Gap in the annotations'
					t2 = stm.split('\n')
					t3 = ' '.join([medianame,'SIL',medianame+'_SIL',t1[0].split(' ')[-1],t1[1].split(' ')[-2]])
					stm = '\n'.join(t2[:-2] + [t3] + t2[-2:])
				else:
					t2 = stm.split('\n')
					t3 = t1[1].split(' ')
                                        t3[3] = t1[0].split(' ')[-1]
					t4 = ' '.join(t3)
					stm = '\n'.join(t2[:-2] + [t4] + [''])
					
		except:
			pass
	stm = stm.replace(' .0 ',' 0.000 ')
	outfile = open(getName(chaFile),'w')
	outfile.write(stm)
	outfile.close()

chafiles = glob.glob('../VanDam/cha/*.cha')
chafiles.remove('../VanDam/cha/FM07_020713d.cha')

for chaFile in chafiles: 
	print chaFile
	makeFixedStm(chaFile)



