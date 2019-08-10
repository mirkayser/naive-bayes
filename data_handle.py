#! /usr/bin/python -W ignore::DeprecationWarning

import sys,os
import math
import numpy as np

class data_handle(object):
	
	"""convenience class to read csv files into dictionaries with numpy arrays"""
	
	def read_files(self,independent=True,dirname='./data/'):
		
		"""looks up all available input files and executes feature specific read functions"""
		
		files = os.listdir(dirname)
		files = [ os.path.join(dirname, f) for f in files ]
		
		# get filenames for specific feature
		age_files,gen_files,os_gen_files,os_dist_files=[],[],[],[]
		for f in files:
			if 'verifiersage' in f: age_files.append(f)
			elif 'femaleverpercage' in f: gen_files.append(f)
			elif 'gender_veros' in f: os_gen_files.append(f)
			elif 'distribution_os' in f: os_dist_files.append(f)
		
		# read features into numpy arrays 
		self.features={}
		
		if independent:
			if len(age_files)>0: self.read_age_files(age_files)
			if len(gen_files)>0: self.read_gen_files(gen_files)
			if len(os_gen_files)>0 and len(os_dist_files)>0: self.read_os_files(age_files,os_gen_files,os_dist_files)
		
		else:
			if len(gen_files)>0: self.read_dependent_gen_files(gen_files)
			if len(os_gen_files)>0 and len(os_dist_files)>0: self.read_dependent_os_files(age_files,os_gen_files,os_dist_files)
			if len(gen_files)==0 and len(os_gen_files)==0 and len(age_files)>0:
				self.read_age_files(age_files)
			
	def read_age_files(self,files):
		
		"""convenience function to read csv files with age data into numpy array"""
		
		for i,f in enumerate(files):
			
			a = np.genfromtxt(f, delimiter=',',names=True,usecols=(1,2,3),dtype=None)

			if i==0: 	array = np.copy(a)
			else:			array = np.concatenate((array,a))
		
		ages,voters,vers,nons,p_vers,p_nons = [],[],[],[],[],[]
		i=18
		while i<=104:
			
			ages.append( i )
			
			m = array['age'] == i
			if np.sum(array[m]['votersx'])>0:
				voters.append( np.sum(array[m]['votersx']) )
				vers.append( np.sum(array[m]['votersy']) )
				nons.append( np.sum(array[m]['votersx']) - np.sum(array[m]['votersy']) )
				p_vers.append( np.sum(array[m]['votersy']) / float(np.sum(array[m]['votersx'])) )
				p_nons.append( ( np.sum(array[m]['votersx']) - np.sum(array[m]['votersy']) ) / float(np.sum(array[m]['votersx'])) )
			else:
				voters.append( voters[-1] )
				vers.append( vers[-1] )
				nons.append( nons[-1] )
				p_vers.append( p_vers[-1] )
				p_nons.append( p_nons[-1] )
			
			i+=1
		
		a = np.array( zip(ages,voters,vers,nons,p_vers,p_nons),dtype=[('f',int),('voters',int),('ver',int),('non',int),('p_v',float),('p_n',float)] )
		self.features['age'] = a
		
	def read_gen_files(self,files):

		"""convenience function to read csv files with gender data into numpy array"""
		
		for i,f in enumerate(files):
			
			a = np.genfromtxt(f, delimiter=',',names=True,usecols=(2,3,6,7),dtype=None)
			
			if i==0: 	array = np.copy(a)
			else:			array = np.concatenate((array,a))
		
		gens,voters,vers,nons = [],[],[],[]
		
		d = {'female':'F','male':'M'}
		
		keys = sorted(d.keys())
		gens,voters,vers,nons,p_vers,p_nons = [],[],[],[],[],[]
		for k in keys:
			
			if np.sum(array[d[k]+'x'])==0: continue
			
			gens.append( k )
			voters.append( np.sum(array[d[k]+'x']) )
			vers.append( np.sum(array[d[k]+'y']) )
			nons.append( np.sum(array[d[k]+'x'])-np.sum(array[d[k]+'y']) )
			p_vers.append( np.sum(array[d[k]+'y']) / float(np.sum(array[d[k]+'x'])) )
			p_nons.append( (np.sum(array[d[k]+'x'])-np.sum(array[d[k]+'y'])) / float(np.sum(array[d[k]+'x'])) )
		
		a = np.array( zip(gens,voters,vers,nons,p_vers,p_nons),dtype=[('f','S10'),('voters',int),('ver',int),('non',int),('p_v',float),('p_n',float)] )
		self.features['gender'] = a
		
	def read_os_files(self,age_files,gen_files,dist_files):
		
		"""convenience function to read csv files with os data into numpy array"""
		
		# sort filenames into tuples
		files = []
		for af in age_files:
			for gf in gen_files:
				for df in dist_files:
					if af.split("_")[0][-4:]==gf.split("_")[0][-4:]==df.split("_")[0][-4:]:
						files.append( (af,gf,df) )
		
		if len(files)!=len(gen_files): 
			raise ValueError("""all three files (*_verifiersage.csv, *_gender_veros.csv, *_distribution_os.csv)
					from the same year have to be in input data""")
		
		for i,f in enumerate(files):
			
			a = np.genfromtxt(f[0], delimiter=',',names=True,usecols=(1,2,3),dtype=None)
			g = np.genfromtxt(f[1], delimiter=',',names=True,usecols=(1,2,3),dtype=None)
			d = np.genfromtxt(f[2], delimiter=',',names=True,usecols=(1,2,3),dtype=None)
			
			# delete quotation in g
			for j in xrange(len(g)):
				g[j]['vos'] = g[j]['vos'].replace('"','') 
			
			# get verified
			for j in xrange(len(g)):
				num = a[a['age']==g[j]['age']]['votersy'][0]
				per = g[j]['proc']/100.
				
				g[j]['proc'] = int(per*num)
			
			# get non-verified
			for j in xrange(len(d)):
				tmp = a[a['age']==d[j]['age']]
				num = tmp['votersx'][0]-tmp['votersy'][0]
				per = d[j]['proc']/100.
				
				d[j]['proc'] = int( math.ceil(per*num) )
			
			if i==0:
				v_array = np.copy(g)
				n_array = np.copy(d)
			else:
				v_array = np.concatenate((v_array,g))
				n_array = np.concatenate((n_array,d))
			
		# generate lists
		os = ['android','iphone','windows']
		voters,vers,nons,p_vers,p_nons = [],[],[],[],[]
		for o in os:
			
			m = v_array['vos']==o
			ver = np.sum( v_array[m]['proc'] )
			non = np.sum( n_array[m]['proc'] )
			
			if ver+non>0:
			
				voters.append( ver+non )
				vers.append( ver )
				nons.append( non )
				p_vers.append( ver / float(ver+non) )
				p_nons.append( non / float(ver+non) )
					
		a = np.array( zip(os,voters,vers,nons,p_vers,p_nons),dtype=[('f','S10'),('voters',int),('ver',int),('non',int),('p_v',float),('p_n',float)] )
		self.features['os'] = a
		
	def read_dependent_gen_files(self,files):

		"""convenience function to read csv files with gender data dependent on age into numpy array"""

		for i,f in enumerate(files):
			
			a = np.genfromtxt(f, delimiter=',',names=True,usecols=(1,2,3,6,7),dtype=None)
			
			if i==0: 	array = np.copy(a)
			else:			array = np.concatenate((array,a))
		
		d = {'female':'F','male':'M'}
		keys = sorted(d.keys())

		features,ages,voters,vers,nons = [],[],[],[],[]
		for k in keys:
			
			i=18
			while i<=104:

				features.append( k )
				ages.append( i )
				
				m = array['age'] == i
				if np.sum(array[m][d[k]+'x'])>0:
					voters.append( np.sum(array[m][d[k]+'x']) )
					vers.append( np.sum(array[m][d[k]+'y']) )
					nons.append( np.sum(array[m][d[k]+'x'])-np.sum(array[m][d[k]+'y']) )
				else:
					voters.append( voters[-1] )
					vers.append( vers[-1] )
					nons.append( nons[-1] )
				
				i+=1
			
		a = np.array( zip(features,ages,voters,vers,nons),dtype=[('f','S10'),('age',int),('voters',int),('ver',int),('non',int)] )
		self.features['gender'] = a

	def read_dependent_os_files(self,age_files,gen_files,dist_files):

		"""convenience function to read csv files with os data dependent on age into numpy array"""

		# sort filenames into tuples
		files = []
		for af in age_files:
			for gf in gen_files:
				for df in dist_files:
					if af.split("_")[0][-4:]==gf.split("_")[0][-4:]==df.split("_")[0][-4:]:
						files.append( (af,gf,df) )
		
		if len(files)!=len(gen_files): 
			raise ValueError("""all three files (*_verifiersage.csv, *_gender_veros.csv, *_distribution_os.csv)
					from the same year have to be in input data""")
		
		for i,f in enumerate(files):
			
			a = np.genfromtxt(f[0], delimiter=',',names=True,usecols=(1,2,3),dtype=None)
			g = np.genfromtxt(f[1], delimiter=',',names=True,usecols=(1,2,3),dtype=None)
			d = np.genfromtxt(f[2], delimiter=',',names=True,usecols=(1,2,3),dtype=None)
			
			# delete quotation in g
			for j in xrange(len(g)):
				g[j]['vos'] = g[j]['vos'].replace('"','') 
			
			# get verified
			for j in xrange(len(g)):
				num = a[a['age']==g[j]['age']]['votersy'][0]
				per = g[j]['proc']/100.
				
				g[j]['proc'] = int( math.ceil(per*num) )
			
			# get non-verified
			for j in xrange(len(d)):
				tmp = a[a['age']==d[j]['age']]
				num = tmp['votersx'][0]-tmp['votersy'][0]
				per = d[j]['proc']/100.
				
				d[j]['proc'] = int( math.ceil(per*num) )
			
			if i==0:
				v_array = np.copy(g)
				n_array = np.copy(d)
			else:
				v_array = np.concatenate((v_array,g))
				n_array = np.concatenate((n_array,d))
		
		# generate lists
		os = ['android','iphone','windows']
		features,ages,voters,vers,nons = [],[],[],[],[]
		for o in os:
			
			m1 = v_array['vos']==o
			
			i=18
			while i<=104:
				
				features.append( o )
				ages.append( i )
				
				m2 = v_array[m1]['age'] == i
				
				ver = np.sum( v_array[m1][m2]['proc'] )
				non = np.sum( n_array[m1][m2]['proc'] )
				
				if ver+non > 0:
					voters.append( ver+non )
					vers.append( ver )
					nons.append( non )
				else:
					voters.append( voters[-1] )
					vers.append( vers[-1] )
					nons.append( nons[-1] )
				
				i+=1
				
		a = np.array( zip(features,ages,voters,vers,nons),dtype=[('f','S10'),('age',int),('voters',int),('ver',int),('non',int)] )
		self.features['os'] = a
