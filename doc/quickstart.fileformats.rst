.. _quickstart.fileformats:

File formats
============

Point and trajectory files
--------------------------

|inferencemap| accepts text files as input data files with |txt|, |xyt|, |trxyt| or any other extension. These text files contain numerical values organized in tab-delimited columns such that each row specifies the euclidean coordinates of the position of a molecule along with time and optionally trajectory number.

Trajectory number, if available, is the first column. Time in |seconds| is always the last columns. All the intermediate columns - usually two (`x`, `y`) or three (`x`, `y`, `z`) - are the coordinates in |um| of the position.

A single dataset can be split in several files.

Tesselation *.imt.h5* files
---------------------------

Subextension |imtalone| is not compulsory but adding any subextension is recommended.
In Python, a |imt| file can be loaded one at a time as follows::

	from inferencemap.io import HDF5Store

	hdf = HDF5Store(path_to_imt_file)
	cells = hdf.peek('cells')
	hdf.close

The tesselation/partition object is ``cells``.

See :class:`~inferencemap.tesselation.CellStats` for further reference information about ``cells`` structure.

Infered map *.map.h5* files
---------------------------

(to do)

.. |txt| replace:: *.txt*
.. |xyt| replace:: *.xyt*
.. |trxyt| replace:: *.trxyt*
.. |imtalone| replace:: *.imt*
.. |imt| replace:: *.imt.h5*
.. |map| replace:: *.map.h5*
.. |seconds| replace:: **seconds**
.. |um| replace:: **µm**
.. |inferencemap| replace:: **InferenceMAP**

