
InferenceMAP's file formats
===========================

`.imt.h5` tesselation file
--------------------------

This file contains a single datatype `cells` which is an HDF5 representation of a 
:class:`inferencemap.tesselation.CellStats` object. Below is a list of key attributes:

* `cells`:

	* `points`: point coordinates (original data; size `n` x `d+`)
	* `cell_index`: index (from `0` to `c - 1`) of the containing cell (size `n`)
	* `cell_count`: number of points per cell (size `c`)
	* etc (to do)
	* `tesselation`:

		* `cell_centers`: centroid coordinates (size `c` x `d`)
		* `cell_adjacency`: CSR sparse matrix representing the adjacency relationship between cells (size `c` x `c`)
		* etc (to do)

