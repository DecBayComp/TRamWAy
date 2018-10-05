
import rwa
from . import *
import importlib
import copy

# 0.2 -> 0.3: typo
translation_table = [
        ('tramway.tesselation.base.CellStats',  'tramway.tessellation.base.CellStats'),
        ('tramway.tesselation.base.Delaunay',   'tramway.tessellation.base.Delaunay'),
        ('tramway.tesselation.base.Voronoi',    'tramway.tessellation.base.Voronoi'),
        ('tramway.tesselation.base.RegularMesh',        'tramway.tessellation.base.RegularMesh'),
        ('tramway.tesselation.nesting.NestedTesselations',      'tramway.tessellation.nesting.NestedTessellations'),
        ('tramway.tesselation.kdtree.KDTreeMesh',       'tramway.tessellation.kdtree.KDTreeMesh'),
        ('tramway.tesselation.kmeans.KMeansMesh',       'tramway.tessellation.kmeans.KMeansMesh'),
        ('tramway.tesselation.gas.GasMesh',     'tramway.tessellation.gas.GasMesh'),
        ('tramway.tesselation.time.TimeLattice',        'tramway.tessellation.time.TimeLattice'),
        ('tramway.tesselation.plugins.window.SlidingWindow',    'tramway.tessellation.plugins.window.SlidingWindow'),
        ]

# 0.2 -> 0.3: new project structure
translation_table += [
        ('tramway.inference.diffusivity.DV',    'tramway.inference.dv.DV'),
        ('tramway.spatial.scaler.Scaler',       'tramway.core.scaler.Scaler'),
        ('tramway.tessellation.base.RegularMesh',       'tramway.tessellation.grid.RegularMesh'),
        ('tramway.spatial.dichotomy.Dichotomy', 'tramway.tessellation.kdtree.dichotomy.Dichotomy'),
        ('tramway.spatial.dichotomy.ConnectedDichotomy',        'tramway.tessellation.kdtree.dichotomy.ConnectedDichotomy'),
        ('tramway.spatial.gas.Gas',     'tramway.tessellation.gwr.gas.Gas'),
        ('tramway.spatial.graph.array.ArrayGraph',      'tramway.tessellation.gwr.graph.array.ArrayGraph'),
        ('tramway.tessellation.gas.GasMesh',    'tramway.tessellation.gwr.GasMesh'),
        ('tramway.tessellation.plugins.window.SlidingWindow',   'tramway.tessellation.window.SlidingWindow'),
        ]

# 0.2 -> 0.3: type refinements
translation_table += [
        ('tramway.inference.base.Cell', 'tramway.inference.base.Translocations'),
        ]


def translate_types(translation_table):
        """Translate types for rwa files.
        """
        # reduce the table
        _table = dict(translation_table)
        _any_update = True
        while _any_update:
                _any_update = False
                for _former_type, _current_type in list(_table.items()):
                        try:
                                _more_current_type = _table[_current_type]
                        except KeyError:
                                pass
                        else:
                                _table[_former_type] = _more_current_type
                                _any_update = True

        # load and register
        for _former_type, _current_type in _table.items():
                _module_name = _current_type.split('.')
                _type_name = _module_name.pop()
                _module_name = '.'.join(_module_name)
                _module = importlib.import_module(_module_name)
                _type = getattr(_module, _type_name)
                try:
                        _storable = rwa.hdf5.hdf5_service.byStorableType(_current_type)
                except:
                        _storable = rwa.default_storable(_type, storable_type=_current_type)
                        _storable_handler = _storable.handlers[0]
                else:
                        _storable_handler = copy.copy(_storable.asVersion())
                _storable_handler._poke = None # peek only
                _storable = rwa.Storable(_type, key=_former_type, handlers=_storable_handler)
                rwa.hdf5_storable(_storable, agnostic=True)

translate_types(translation_table)

