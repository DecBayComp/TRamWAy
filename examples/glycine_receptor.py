
import os
#from pathlib import Path
#import urllib.request # only in PY3
from inferencemap.helper.tesselation import *

#demo_name = 'glycine-receptor'

short_description = 'various tesselations of glycine receptor data'

#data_server = 'http://157.99.149.144/'
data_server = 'http://dl.pasteur.fr/fop/jod5vvB5/'
data_file = 'glycine_receptor.trxyt'
data_dir = ''

if __name__ == '__main__':
	main()

def main():
	local = os.path.join(data_dir, data_file)

	## download data file if missing
	#if not os.path.exists(local):
	#	urllib.request.urlretrieve(data_server + data_file, local)

	output_basename, _ = os.path.splitext(local)
	def out(method, extension):
		return '{}.{}{}'.format(output_basename, method, extension)

	# gwr
	print("\nfourth approach: GWR + knn=40 + overlap")
	method = 'gwr'
	gwr = tesselate(local, method, output_file=out(method, '.h5'), \
		verbose=True)
	cell_plot(gwr, output_file=out(method, '.png'), show=True, \
		verbose=True)


