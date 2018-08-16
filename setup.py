# -*- coding: utf-8 -*-

from setuptools import setup
from codecs import open
from os import path

# requirements moved to requirements.txt
install_requires = []#'six', 'numpy', 'scipy', 'pandas', 'tables', 'matplotlib', 'rwa-python>=0.7.1'
extras_require = {}
setup_requires = ['pytest-runner']
tests_require = ['pytest']


pwd = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(pwd, 'README.rst'), encoding='utf-8') as f:
        long_description = f.read()

setup(
        name = 'tramway',
        version = '0.3-rc8',
        description = 'TRamWAy',
        long_description = long_description,
        url = 'https://github.com/DecBayComp/TRamWAy',
        author = 'François Laurent',
        author_email = 'francois.laurent@pasteur.fr',
        license = 'CeCILL v2.1',
        classifiers = [
                'Intended Audience :: Science/Research',
                'Programming Language :: Python :: 2',
                'Programming Language :: Python :: 2.7',
                'Programming Language :: Python :: 3',
                'Programming Language :: Python :: 3.5',
                'Programming Language :: Python :: 3.6',
        ],
        keywords = '',
        package_dir = {'tramway': 'tramway'},
        packages = ['tramway',
                'tramway.core',
                'tramway.core.analyses',
                'tramway.core.hdf5',
                'tramway.tessellation',
                'tramway.tessellation.grid',
                'tramway.tessellation.gwr',
                'tramway.tessellation.gwr.graph',
                'tramway.tessellation.kdtree',
                'tramway.tessellation.kmeans',
                'tramway.inference',
                'tramway.feature',
                'tramway.plot',
                'tramway.plot.tk',
                'tramway.helper',
                'tramway.helper.simulation',
                'tramway.utils'],
        scripts = ['scripts/tramway'],
        install_requires = install_requires,
        extras_require = extras_require,
        setup_requires = setup_requires,
        tests_require = tests_require,
        package_data = {},
)
