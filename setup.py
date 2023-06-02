# -*- coding: utf-8 -*-

from setuptools import setup
from codecs import open
from os import path

# requirements moved to requirements.txt
install_requires = ['six', 'numpy', 'scipy', 'pandas', 'matplotlib', 'rwa-python>=0.8']
extras_require = {
        'animate':  ['opencv-python', 'scikit-image', 'tqdm'],
        'roi':  ['polytope', 'tqdm'],
        'webui':    ['bokeh >=2.0.2, <2.3.0', 'selenium', 'plotly', 'nbformat'],
        'hpc-minimal':  ['polytope', 'stopit'],
        'hpc':  ['polytope', 'paramiko', 'stopit', 'nbconvert'],
        'full':  [
            'polytope', 'paramiko', 'stopit', 'nbconvert',
            'bokeh >=2.0.2, <2.3.0', 'selenium', 'plotly', 'nbformat',
            'opencv-python', 'scikit-image', 'tqdm',
            ]}
tests_require = ['pytest']


pwd = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(pwd, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = 'tramway',
    version = '0.6.6',
    description = 'TRamWAy',
    long_description = long_description,
    url = 'https://github.com/DecBayComp/TRamWAy',
    author = 'Institut Pasteur',
    author_email = 'francois.laurent@pasteur.fr',
    license = 'CECILL-2.1',
    classifiers = [
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords = '',
    package_dir = {'tramway': 'tramway'},
    packages = ['tramway',
        'tramway.core',
        'tramway.core.analyses',
        'tramway.core.hdf5',
        'tramway.core.parallel',
        'tramway.tessellation',
        'tramway.tessellation.grid',
        'tramway.tessellation.gwr',
        'tramway.tessellation.gwr.graph',
        'tramway.tessellation.kdtree',
        'tramway.tessellation.kmeans',
        'tramway.inference',
        'tramway.inference.bayes_factors',
        'tramway.localization',
        'tramway.localization.UNet',
        'tramway.tracking',
        'tramway.tracking.track_non_track',
        'tramway.feature',
        'tramway.feature.single_traj',
        'tramway.plot',
        'tramway.plot.animation',
        'tramway.plot.bokeh',
        'tramway.plot.tk',
        'tramway.helper',
        'tramway.helper.inference',
        'tramway.helper.simulation',
        'tramway.utils',
        'tramway.analyzer',
        'tramway.analyzer.attribute',
        'tramway.analyzer.artefact',
        'tramway.analyzer.spt_data',
        'tramway.analyzer.roi',
        'tramway.analyzer.time',
        'tramway.analyzer.tesseller',
        'tramway.analyzer.tesseller.post',
        'tramway.analyzer.sampler',
        'tramway.analyzer.mapper',
        'tramway.analyzer.env',
        'tramway.analyzer.browser',
        'tramway.analyzer.pipeline',
        'tramway.analyzer.images',
        'tramway.analyzer.localizer',
        'tramway.analyzer.tracker',
        ],
    scripts = ['scripts/tramway', 'scripts/tramway-browse'],
    install_requires = install_requires,
    extras_require = extras_require,
    tests_require = tests_require,
    package_data = {},
)
