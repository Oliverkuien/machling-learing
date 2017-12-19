import sys
import os
import numpy as np

def loaddataset():
 listword = [['my','dog','has','flea','problems','help','please'],
	     ['maybe','not','take','him','to','dog','park','stupid'],
	     ['my','dalmation','is','so','cute','I','love','him','my'],
	     ['stop','posting','stupid','worthless','garbage'],
	     ['my','licks','ate','my','steak','how','to','stop','him'],
	     ['quit','buying','worthless','dog','food','stupid']]
 classlabel = [0,1,0,1,0,1]
 return listword,classlabel

class Bayes(object):
 def __init__(self):
	self.vocabulary = []
	self.idf = 0
	self.tf = 0
	self.tdm = 0
	self.pcates = {}
	self.doclength = 0
	self.vocablen = 0
	
 def prob(self,classlabel):
	self.labels = classlabel
	labeltemps = set(self.labels)
	for labeltemp in labeltemps:
		self.pcates[labeltemp] = float(self.labels.count(labeltemp)) / float(len(self.labels))
 def word(self,trainset):
	self.idf = np.zeros([1,self.vocablen])
	self.tf = np.zeros([self.doclength,self.vocablen])
	for line in  xrange(self.doclength):
	 for word in trainset[line]:
		self.tf[line,self.vocabulary.index(word)] += 1
	 for signleword in set(trainset[line]) :
		self.idf[0,self.vocabulary.index(signleword)] += 1
 def build_tdm(self):
	self.tdm = np.zeros([len(self.pcates),self.vocablen])
	sumlist = np.zeros([len(self.pcates),1])
	for line in xrange(self.doclength):
		self.tdm[self.labels[line]] += self.tf[line]
		sumlist[self.labels[line]] = np.sum(self.tdm[self.labels[line]])
	self.tdm = self.tdm/sumlist
 def predict(self,testset):
	a = testset
	testset = np.zeros([1,31])
	#print self.vocabulary
	for word in a:
	 #print word
	 testset[0,self.vocabulary.index(word)] += 1	
	predict = 0
	pred = ''
	for vect,preclass in zip(self.tdm,self.pcates):
		temp = np.sum(testset*vect*self.pcates[preclass])
		if temp > predict:
		 predict = temp
		 pred = preclass
	return pred
 def train_set(self,trainset,classlabel):
	self.prob(classlabel)
	self.doclength = len(trainset)
	tempset = set()
	[tempset.add(word) for doc in trainset for word in doc]
	self.vocabulary = list(tempset)
	self.vocablen = len(self.vocabulary)
	#print self.vocablen
	self.word(trainset)
	self.build_tdm()
	
listword,classlabel = loaddataset()
nb=Bayes()
nb.train_set(listword,classlabel)
b=['my','dog','is','cute','I','love','dog']
print b
if nb.predict(b) == 0 :
	print 'it is good news' 
else :
	print 'it is negative '
	
	
   
