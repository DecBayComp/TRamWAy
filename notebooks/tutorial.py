
import glob
import os.path
from math import *
import numpy as np
import tempfile
from tramway.helper import *
from tramway.helper.simulation import *

np.random.seed(123456789)

tutorials_dir = os.path.dirname(__file__)
tutorial_data_dir = os.path.join(tutorials_dir, 'data-examples')

def _exists(f):
    try:
        return 0 < os.stat(f).st_size
    except FileNotFoundError:
        return False

def download_RWAnalyzer_tour_data(extra=False):
    """
    Checks for the availability of the *demo1.rwa* file
    and downloads all the data files if missing.
    """
    any_data_file = os.path.join(tutorial_data_dir, 'demo1.rwa')
    if not os.path.isfile(any_data_file):
        if extra:
            data_archive = 'https://gitlab.pasteur.fr/flaurent/tramway-artefacts/-/raw/main/notebooks/RWAnalyzer_tour_data_extra.tar.bz2?inline=false'
        else:
            data_archive = 'https://gitlab.pasteur.fr/flaurent/tramway-artefacts/-/raw/main/notebooks/RWAnalyzer_tour_data.tar.bz2?inline=false'
        tutorial_data_file = data_archive.split('/')[-1].split('?')[0]
        try:
            from urllib.request import urlretrieve
        except: # Python2
            from urllib import urlretrieve
        urlretrieve(data_archive, tutorial_data_file)
        import tarfile
        with tarfile.open(tutorial_data_file) as archive:
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
                tar.extractall(path, members, numeric_owner=numeric_owner)
            safe_extract(archive, os.path.dirname(tutorial_data_dir))

def print_analysis_tree(analyses, annotations=False, **kwargs):
    """
    Prints the analysis tree, just like ``print(analyses)``, with options.
    """
    if annotations is True:
        import tramway.tessellation.base as tessellation
        import tramway.inference.base as inference
        def annotate(label, _type):
            if label == '':
                return 'original SPT data'
            elif issubclass(_type, tessellation.Partition):
                return 'data sampling'
            elif issubclass(_type, inference.Maps):
                return 'parameter maps'
        annotations = annotate
    if 'node' not in kwargs:
        import rwa
        kwargs['node'] = rwa.lazytype
    print(format_analyses(analyses, annotations=annotations, **kwargs))


