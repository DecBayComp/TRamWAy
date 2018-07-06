# -*- coding: utf-8 -*-

# Copyright © 2018, Institut Pasteur
#   Contributor: Alexander Serov

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from .calculate_bayes_factors import calculate_bayes_factors

# Only this function is supposed to be used publicly.
# The package can be imported by just `import bayes_factors`.
__all__ = ['calculate_bayes_factors']
