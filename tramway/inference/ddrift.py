# -*- coding: utf-8 -*-

# Copyright © 2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import tramway.inference.degraded_ddrift as degraded_dd
import tramway.inference.standard_ddrift as standard_dd


setup = dict(standard_dd.setup) # copy
setup['name'] = ('dd', 'ddrift')
del setup['provides']
setup['infer'] = 'infer_DD'


def infer_DD(cells, diffusivity_prior=None, drift_prior=None, jeffreys_prior=False,
        min_diffusivity=None, max_iter=None, epsilon=None, rgrad=None, **kwargs):

    if diffusivity_prior is None and drift_prior is None:
        return degraded_dd.infer_DD(cells, jeffreys_prior=jeffreys_prior,
                min_diffusivity=min_diffusivity, **kwargs)
    else:
        return standard_dd.infer_smooth_DD(cells, diffusivity_prior, drift_prior, jeffreys_prior,
                min_diffusivity, max_iter, epsilon, rgrad, **kwargs)

