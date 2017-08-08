

f=open('total','r')

f = f.read().split('\n')
s = 0
c = 0
for row in f:
	if 'F1' in row:
		s+=float(row[4:])
		c+=1

print 'F1 ',
print s/c



s = 0
c = 0
for row in f:
        if 'Precision' in row:
                s+=float(row[10:])
                c+=1

print 'Precision ',
print s/c


s = 0
c = 0
for row in f:
        if 'Recall' in row:
                s+=float(row[8:])
                c+=1

print 'Recall ',
print s/c


