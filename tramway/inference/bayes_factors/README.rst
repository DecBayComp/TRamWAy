This is a module for calculation of the Bayes factors.

Dependence: tqdm

Usage:

import bayes_factors # load the module

Bs, forces, min_ns = calculate_bayes_factors(zeta_ts, zeta_sps, ns, Vs, Vs_pi, B_threshold)

Input:

zeta_ts --- signal-to-noise ratios for the total force in individual bins = dx_mean / sqrt(var(dx)) in bins. Size: M x 2,
zeta_sps --- signal-to-noise ratios for the spurious force in individual bins = grad(D) dt / sqrt(var(dx)) in bins. Size: M x 2,
ns --- number of jumps in each bin. Size: M x 1,
Vs --- jump variance in individual bins = E((dx - dx_mean) ** 2). Size: M x 1,
Vs_pi --- jump variance in all other bins but the current bin for each bin. Size: M x 1,
B_threshold --- required level of evidence for thresholding the Bayes factor. The Bayes factor will be thresholded at B_threshold and 1/B_threshold. Scalar. Default: 10

Output:

Bs --- Bayes factor values in the bins. Size: M x 1,
forces --- returns 1 if there is strong evidence for the presence of a conservative forces, -1 if strong evidence for a spurious force, and 0 if evidence is insufficient. Size: M x 1.

Notation:
M --- number of bins.
