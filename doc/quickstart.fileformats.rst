.. _quickstart.fileformats:

File formats
============

Tesselation *.imt.h5* files
---------------------------

Subextension ``.imt`` is not compulsory but a subextension (any) is recommended.

In Python, they can be loaded as follows::

	from inferencemap.io import HDF5Store

	hdf = HDF5Store(path_to_imt_file)
	cells = hdf.peek('cells')
	hdf.close()

The tesselation/partition object is ``cells``.

See :class:`~inferencemap.tesselation.CellStats` for further reference information about ``cells`` structure.

