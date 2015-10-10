import pickle
import gzip
import numpy

#open files
with open('s01.dat','rb') as f:
	p = pickle._Unpickler(f)
	p.encoding= ('latin1')
	x = p.load()
	print(x)
