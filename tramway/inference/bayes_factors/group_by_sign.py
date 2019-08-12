# TODO: clean-up the file
import sys
from collections import deque  # a standard FIFO queue

import numpy as np


def group_by_sign(cells, tqdm, B_threshold=10, random_group_order=True, **kwargs):
    """
    Parameters:
    lg_Bs,   nd-array of Bayes factor values of individual cells, Nx1
    adjacency,  cell adjacency matrix, sparse, NxN

    Return:

    """

    # Prepare the data. A Cell is a CellStats object
    adjacency = cells.adjacency
    N = cells.adjacency.shape[0]
    lg_Bs = np.full(N, np.nan)
    for key in cells:
        lg_Bs[key] = cells[key].lg_B
    lg_Bs = np.concatenate(lg_Bs, axis=None)

    N = len(lg_Bs)
    # Initialize queues
    # remaining_queue = deque(range(N))
    queue = deque()
    finished = set()
    groups = np.full(N, np.nan)

    signs = np.sign(lg_Bs)

    def process_one_cell(i, group):
        """If cell's neighbors have same sign, add to active queue"""

        # Mark cell
        finished.add(i)
        groups[i] = group

        neighbs = adjacency.getrow(i).nonzero()[1]
        sign = signs[i]
        for n in neighbs:
            if signs[n] == sign:
                queue.append(n)

    # Main cycle
    group = 0
    for i in tqdm(range(N), desc='Grouping force regions'):
        if i in finished:
            continue

        # No group if nan
        if np.isnan(signs[i]):
            continue

        queue.append(i)
        while queue:
            j = queue.popleft()
            if j not in finished:
                process_one_cell(j, group)
        group += 1
    max_group = group - 1

    # Groups identified. Now calculate the Bayes factors of the groups, and the thresholded maps
    group_lg_B = np.full(N, np.nan)
    for group in range(max_group):
        inds = groups == group
        group_lg_B[inds] = lg_Bs[inds].sum()

    # Threshold the results at the given significance level
    group_forces = 1 * (group_lg_B >= np.log10(B_threshold)) - \
        1 * (group_lg_B <= -np.log10(B_threshold))

    # Randomize group order
    if random_group_order:
        new_groups = np.random.permutation(max_group + 1)
        for i in range(len(groups)):
            if ~np.isnan(groups[i]):
                groups[i] = new_groups[int(groups[i])]

    # Store data in cells
    for key in cells:
        cells[key].groups = groups[key]
        cells[key].group_lg_B = group_lg_B[key]
        cells[key].group_forces = group_forces[key]

    # print(groups, group_lg_B, group_forces)
    # sys.exit(1)
    # raise RuntimeError('Manual stop')
