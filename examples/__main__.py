#!/usr/bin/env python

from __future__ import print_function

import re
import os
import sys
import six
import argparse
if six.PY2:
	#import importlib2 as importlib
	#importlib2.hook.install()
	import urllib as request
	fullmatch = re.match
else:
	import urllib.request as request
	fullmatch = re.fullmatch
import importlib
#from pathlib import Path


def run(module):
	def __main__(args):
		if hasattr(module, 'data_file'):
			if hasattr(module, 'data_dir'):
				data_dir = module.data_dir
			else:
				data_dir = ''
			local = os.path.join(module.data_dir, module.data_file)
			if not os.path.exists(local) and hasattr(module, 'data_server'):
				print('downloading {}... '.format(module.data_file), end='')
				try:
					request.urlretrieve(os.path.join(module.data_server, \
							module.data_file), local)
					print('[done]')
				except:
					print('[failed]')
		args = args.__dict__
		del args['func']
		module.main(**args)
	return __main__


def main():
	demo_dir = os.path.dirname(__file__)
	demo_path = __package__
	pattern = re.compile(r'[a-zA-Z0-9].*[.]py')
	candidate_demos = [ os.path.splitext(fn)[0] \
		for fn in os.listdir(demo_dir) \
		if fullmatch(pattern, fn) is not None ]
	demos = {}
	for demo in candidate_demos:
		path = '{}.{}'.format(demo_path, demo)
		if True:#six.PY2:
			if True:#try:
				module = importlib.import_module(path)
				if hasattr(module, 'demo_name'):
					demo = module.demo_name
				else:
					demo = demo.replace('_', '-')
				demos[demo] = module
			#except ImportError:
			#	pass
		else:
			spec = importlib.util.find_spec(path)
			if spec is not None:
				#print('importing {}...'.format(demo))
				module = importlib.util.module_from_spec(spec)
				spec.loader.exec_module(module)
				if hasattr(module, 'demo_name'):
					demo = module.demo_name
				else:
					demo = demo.replace('_', '-')
				demos[demo] = module

	parser = argparse.ArgumentParser(prog='tramway-demo', \
		description='TRamWAy demo launcher.', \
		epilog='See also https://github.com/DecBayComp/TRamWAy')
	#parser.add_argument('demo', choices=demos.keys())
	hub = parser.add_subparsers(title='demos', \
		description="type '%(prog)s demo-name --help' for additional help")
	for demo in demos:
		module = demos[demo]
		if hasattr(module, 'short_description'):
			short_help = module.short_description
		else:
			short_help = None
		demo_parser = hub.add_parser(demo, help=short_help)
		if hasattr(module, 'arguments'):
			for args, kwargs in module.arguments:
				if not isinstance(args, (tuple, list)):
					args = (args,)
				try:
					demo_parser.add_argument(*args, **kwargs)
				except:
					print('failed to add argument: {}'.format(args))
		demo_parser.set_defaults(func=run(module))

	# parse
	args = parser.parse_args()
	#module = demos[args.demo]
	if not hasattr(args, 'func'):
		parser.print_help()
		parser.exit(0)
	args.func(args)


if __name__ == '__main__':
	main()


