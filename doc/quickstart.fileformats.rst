.. _quickstart.fileformats:

File formats
============

Point and trajectory files
--------------------------

|tramway| accepts text files as input data files with |txt|, |xyt|, |trxyt| or any other extension. These text files contain numerical values organized in tab-delimited columns such that each row specifies the euclidean coordinates of the position of a molecule along with time and optionally trajectory number.

Trajectory number, if available, is the first column. Time in |seconds| is always the last columns. All the intermediate columns - usually two (`x`, `y`) or three (`x`, `y`, `z`) - are the coordinates in |um| of the position.

A single dataset can be split in several files.

Tesselation *.imt.rwa* files
---------------------------

Subextension |imtalone| is not compulsory.
In Python, a |imt| file can be loaded one at a time as follows::

	from tramway.io import HDF5Store

	hdf = HDF5Store(path_to_rwa_file)
	cells = hdf.peek('cells')
	hdf.close

The tesselation/partition object is ``cells``.

See :class:`~tramway.tesselation.CellStats` for further reference information about ``cells`` structure.

Infered map *.map.rwa* files
---------------------------

(to do)

.. |txt| replace:: *.txt*
.. |xyt| replace:: *.xyt*
.. |trxyt| replace:: *.trxyt*
.. |imtalone| replace:: *.imt*
.. |imt| replace:: *.imt.rwa*
.. |map| replace:: *.map.rwa*
.. |seconds| replace:: **seconds**
.. |um| replace:: **µm**
.. |tramway| replace:: **TRamWAy**

