#! /usr/bin/python -W ignore::DeprecationWarning

import sys,os
import math
import numpy as np
import pandas as pd


class NB1():

	"""Naive Bayes Classifer for completely independent features"""

	def __init__(self):
		"""initialize two classes"""
		self.classes = np.array([0,1])

	def get_class_probs(self,dataset):
		"""calculate probability to encounter class k in dataset"""
		
		class_probs=[]
		for k in self.classes:
			
			p=[]
			for fk in dataset.keys():
				
				t = np.sum(dataset[fk]['voters'])
				v = np.sum(dataset[fk]['ver']) 
				n = np.sum(dataset[fk]['non']) 
				
				if k==0:		p.append( n / float(t) )
				elif k==1:	p.append( v / float(t) )
				
			class_probs.append( np.average(p) )
		return class_probs

	def fit(self,dataset,class_probs=None):
		"""train classifier to statistical data in dataset.
		creates array with the probabilities of all available features
		and a dictionary to connect features to their index in the array."""
		
		if class_probs==None: class_probs = self.get_class_probs(dataset)
		self.class_probs = np.array(class_probs)
		
		# feature probs
		self.fdic,index,f_probs = {},0,[]
		for k in self.classes:
			ps = [] 
			for fk in sorted(dataset.keys()):
				for item in dataset[fk]:
					
					if k==0:
						p = item['non'] / float(item['voters'])
						self.fdic[item[0]] = index 
						index+=1
					elif k==1:
						p = item['ver'] / float(item['voters'])
					
					if float(item['voters'])==0: p=np.nan 
					
					ps.append( p )
			f_probs.append( np.array(ps) )
		self.f_probs = np.array(f_probs)

	def get_probabilities(self,classnm):
		return self.f_probs[classnm]

	def get_features(self,data):
		"""returns a feature vector x,
		with value x_i=1 if feature is present and x_i=0 if feature is not present"""
		
		features = []
		for item in data:
			
			tmp = [0]*len(self.fdic.keys())
			for f in item:
				if self.fdic.has_key(f):
					index = self.fdic[f]
					tmp[index] = 1
			features.append(np.array(tmp))
		
		return features

	def get_prob_multinomial(self,features):
		"""returns probabilities to find class k in any sample for all given samples.
		probabilities are calculated using the multinomial event model"""
		
		ps=[]
		for f in features:

			tmp=[]
			for k in self.classes:
			
				pk = self.f_probs[k]
				log_p = np.log( self.class_probs[k] ) + np.sum( f*np.log(pk) )
				p = np.exp(log_p)

				tmp.append(p)
			
			s = np.sum(tmp)
			ps.append( np.array(tmp)/s )

		return np.array(ps)

	def predict(self,data,threshold=False):
		"""assign class k to any sample in data.
		By default the class k with the greater probability is returned.
		If threshold is set to True, class 0 is returned for all samples with p_0 >= median(all samples).
		Else class 1 is returned"""
		
		features = self.get_features(data)

		# get probabilities
		ps = self.get_prob_multinomial(features)

		# calculate median of class 0
		med = np.median(ps[:,0][np.isnan(ps[:,0])==False])
		
		# predict class
		pred = []
		for item in ps:
			try:
				if threshold:
					# select class 0/1 below or above median, respectively
					if item[0]>=med:	p=0
					else:						p=1
				else:
					# select argmax
					p = np.where( item==np.max(item) )[0][0]
				
			except: p=-1
				
			pred.append( p )

		return pred,ps


class NB2():
	
	"""Naive Bayes Classifer for independent features with explicit dependence on another variable (NOT one of the features)"""

	def __init__(self):
		"""initialize two classes"""
		self.classes = np.array([0,1])

	def get_class_probs(self,dataset):
		"""calculate probability to encounter class k in dataset"""
		class_probs=[]
		for k in self.classes:
			
			p=[]
			for fk in dataset.keys():
				
				t = np.sum(dataset[fk]['voters'])
				v = np.sum(dataset[fk]['ver']) 
				n = np.sum(dataset[fk]['non']) 
				
				if k==0:		p.append( n / float(t) )
				elif k==1:	p.append( v / float(t) )
				
			class_probs.append( np.average(p) )
		return class_probs

	def fit(self,dataset,class_probs=None):
		"""train classifier to statistical data in dataset.
		creates pandas panel with the probabilities of all available features.
		items -> classes
		major axis -> features
		minor axis -> age"""
		
		if class_probs==None: class_probs = self.get_class_probs(dataset)
		self.class_probs = np.array(class_probs)
		
		# feature probs
		d = []
		for k in self.classes:
			d.append( [] )
			
			for fk in sorted(dataset.keys()):
			
				for i,f in enumerate(sorted(set(dataset[fk]['f']))):
					d[k].append( [] )
					
					m = dataset[fk]['f'] == f
					for item in dataset[fk][m]:
						
						if float(item['voters'])==0: p=np.nan 
						elif k==0:		p = item['non'] / float(item['voters'])
						elif k==1:		p = item['ver'] / float(item['voters'])
						
						d[k][-1].append(p)
		
		minor = sorted(list(set(dataset[fk]['age'])))
		major = []
		for fk in sorted(dataset.keys()):
			major += sorted(list(set(dataset[fk]['f'])))
		
		self.f_probs = pd.Panel(np.array(d),items=self.classes,major_axis=major,minor_axis=minor)

	def get_probabilities(self,classnm):
		return self.f_probs[classnm]

	def get_prob_multinomial(self,data):
		"""returns probabilities to find class k in any sample for all given samples.
		probabilities are calculated using the multinomial event model"""
		
		ps=[]
		for item in data:
		
			tmp=[]
			for k in self.classes:
				
				pk=[]
				if item[1] in self.f_probs.major_axis: pk.append( self.f_probs.loc[ k,item[1],item[0] ] )
				if item[2] in self.f_probs.major_axis: pk.append( self.f_probs.loc[ k,item[2],item[0] ] )
				
				log_p = np.log( self.class_probs[k] ) + np.sum( np.log(pk) )
				p = np.exp(log_p)

				tmp.append(p)
			
			s = np.sum(tmp)
			ps.append( np.array(tmp)/s )

		return np.array(ps)

	def predict(self,data,threshold=False):
		"""assign class k to any sample in data.
		By default the class k with the greater probability is returned.
		If threshold is set to True, class 0 is returned for all samples with p_0 >= median(all samples).
		Else class 1 is returned"""

		# get probabilities
		ps = self.get_prob_multinomial(data)
		
		# calculate median of class 0
		med = np.median(ps[:,0][np.isnan(ps[:,0])==False])
		
		# predict class
		pred = []
		for item in ps:
			
			if np.isnan(item[0]):	p=-1
			
			elif threshold:	# select class 0/1 below or above median, respectively
				if item[0]>=med:	p=0
				else:						p=1
			
			else:	# select argmax
				p = np.where( item==np.max(item) )[0][0]
				
			pred.append( p )

		return pred,ps
