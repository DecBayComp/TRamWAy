# -*- coding: utf-8 -*-

# Copyright © 2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import tramway.inference.degraded_d as degraded_d
import tramway.inference.standard_d as standard_d


setup = dict(standard_d.setup) # copy
setup['name'] = 'd'
del setup['provides']
setup['infer'] = 'infer_D'


def infer_D(cells, diffusivity_prior=None, jeffreys_prior=None, min_diffusivity=None,
        max_iter=None, epsilon=None, rgrad=None, **kwargs):

    if diffusivity_prior is None:
        return degraded_d.infer_D(cells, jeffreys_prior=jeffreys_prior,
                min_diffusivity=min_diffusivity, **kwargs)
    else:
        return standard_d.infer_smooth_D(cells, diffusivity_prior, jeffreys_prior,
                min_diffusivity, max_iter, epsilon, rgrad, **kwargs)

