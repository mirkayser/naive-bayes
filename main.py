#! /usr/bin/python -W ignore::DeprecationWarning

import sys,os
import math
from optparse import OptionParser
import numpy as np

from data_handle import data_handle
from nbclassifier import *

def read_sample(fname):
	"""reads test samples from csv"""
	return np.genfromtxt(fname, delimiter=',',dtype=None)

def get_classifier(features):
	"""return classifier NB1 or NB2 depending on features"""
	if 'age' in features[features.keys()[0]].dtype.names:
		nb = NB2()
	else:
		nb = NB1()
	return nb

def main():
	print ''

	#parse options
	parser = OptionParser()
	parser.add_option("-i","--independent", dest="independent", action="store_true", default=False,
											help="""NB treats all three variables (age,gender,os) as independent,
											 if not set, gender and os are treated dependent to age (gender(age),os(age))""")
	parser.add_option("-t","--threshold", dest="threshold", action="store_true", default=False,
											help="""NB assigns class 0 or 1 depending on p(sample) below or above threshold, respectively""")
	(options, args) = parser.parse_args()
	
	dh = data_handle()
	dh.read_files(independent=options.independent)
	
	nb = get_classifier(dh.features)
	nb.fit(dh.features,class_probs=[0.5,0.5])
	
	sample = read_sample('sample.csv')
	pred,probs = nb.predict(sample,threshold=options.threshold)
	
	out='Multinomial Naive Bayes Classifier:\n\nclass 0 -> non-verifier\nclass 1 -> verifier\n\n'
	for i in xrange(len(pred)):
		out += '  class={:2d}  p=[{:.5E},{:.5E}]  sample={}\n'.format(pred[i],probs[i][0],probs[i][1],str(sample[i]))
	out += '\n  median class 0: {:.5E}\n\n'.format(np.median(probs[:,0][np.isnan(probs[:,0])==False]))
	print out
	
	# print probabilities of class 0
	f_probs = nb.get_probabilities(0)
	#pd.set_option('display.max_columns',None) #uncomment this line to print complete dataset
	print 'class 0:\n\n',f_probs
	
	if not options.independent and dh.features.has_key('os'):
		# print total voters in os
		print '\n\ntotal voters os:'
		os = dh.features['os']
		for i in [18,19,20,21]:
			print i,np.sum(os[os['age']==i]['voters'])
	
	if options.independent and dh.features.has_key('age'):
		# print total voters in age
		print '\n\ntotal voters age:'
		age = dh.features['age']
		for i in [18,19,20,21]:
			print i,np.sum(age[age['f']==i]['voters'])		
	
if __name__ == "__main__":
	main()
