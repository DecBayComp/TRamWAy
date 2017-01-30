# -*- coding: utf-8 -*-

## see https://packaging.python.org/distributing/#setup-py

from setuptools import setup, find_packages
from codecs import open
from os import path

install_requires = ['six', 'numpy', 'scipy', 'pandas', 'tables', 'h5py']
extras_require = {} 


# mocking out h5py for readthedocs to successfully compile the project and generate the doc
import sys
mock_ok = True
try:
	from unittest.mock import MagicMock
except ImportError as e:
	try:
		from mock import Mock as MagicMock
	except ImportError:
		print(e)
		mock_ok = False

if mock_ok:
	class Mock(MagicMock):
		@classmethod
		def __getattr__(cls, name):
			return MagicMock()

	MOCK_MODULES = ['h5py']
	sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)
	for mod_name in MOCK_MODULES:
		install_requires.remove(mod_name)


pwd = path.abspath(path.dirname(__file__))

## Get the long description from the README file
#with open(path.join(pwd, 'README.rst'), encoding='utf-8') as f:
#	long_description = f.read()

setup(
	name = 'inferencemap',
	version = '0.1',
	description = 'InferenceMAP',
	#long_description = long_description,
	url = 'https://github.com/influencecell/inferencemap',
	author = 'François Laurent',
	author_email = 'francois.laurent@pasteur.fr',
	license = 'MIT',
	classifiers = [
		'Programming Language :: Python :: 2',
		'Programming Language :: Python :: 2.7',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3.5',
	],
	keywords = '',
	packages = find_packages(exclude=['doc', 'examples']),
	install_requires = install_requires,
	extras_require = extras_require,
	package_data = {},
	entry_points = {},
)
