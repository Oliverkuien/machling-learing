import numpy as np
import Image
from pylab import *
import sys
print 'learning picture'
def sigma(x,deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))
n=0
np.random.seed(1)
wo=2*np.random.random((4032,1))-1
wt=2*np.random.random((3024,1))-1
a=0
src=sys.argv[1]
with open (src,'r') as f:
	list = f.readlines()
while n<100 :
	for i in list:
		name , label = i.strip('\n').split(' ')
		y = int(label)
		
		x = asarray(Image.open(name).convert('L'),dtype='float64')	
		l1 = sigma(np.dot(x,wo))
		#print l1.shape
		l2 = sigma(np.dot(l1.T,wt))
		#print l2.shape
		miss2 = y-l2
		l2_delta = miss2*sigma(l2,True)
		wt += np.dot(l1,l2_delta)*0.1
		#print wt
		miss1 = np.dot(wt,l2_delta)
		#print miss1.shape
		l1_delta = miss1*sigma(l1,True)
		#print l1_delta.shape
		wo += np.dot(x.T,l1_delta)*0.0002
		a += 1
		print a
	n += 1
while 1==1:
	print 'tell me you want predict'
	name = raw_input()
	Target = array(Image.open(name).convert('L'))
	l1 = sigma(np.dot(Target,wo))
	l2 = sigma(np.dot(l1.T,wt))
	print l2

	

