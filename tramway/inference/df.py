# -*- coding: utf-8 -*-

# Copyright © 2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import tramway.inference.degraded_df as degraded_df
import tramway.inference.standard_df as standard_df


setup = dict(standard_df.setup) # copy
setup['name'] = 'df'
del setup['provides']
setup['infer'] = 'infer_DF'


def infer_DF(cells, diffusivity_prior=None, force_prior=None, jeffreys_prior=False,
        min_diffusivity=None, max_iter=None, epsilon=None, rgrad=None, **kwargs):

    if diffusivity_prior is None and force_prior is None:
        return degraded_df.infer_DF(cells, jeffreys_prior=jeffreys_prior,
                min_diffusivity=min_diffusivity, **kwargs)
    else:
        return standard_df.infer_smooth_DF(cells, diffusivity_prior, force_prior, None,
                jeffreys_prior, min_diffusivity, max_iter, epsilon, rgrad, **kwargs)

