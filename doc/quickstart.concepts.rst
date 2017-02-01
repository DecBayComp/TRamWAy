.. _quickstart.concepts:

Concepts
========

|inferencemap| generates maps of the dynamic parameters that dictate the motion of single molecules. These parameters may include the diffusion, forces (directional biases), interaction (potential) energies and drift.

|inferencemap| takes molecule locations or trajectories data as input.

A preliminary processing step before generating maps consists of segmentating space and time into cells that accomodate enough molecule locations so that the inference is made possible and, on the hand, are small enough so that relevant spatial(-temporal) variations in the dynamic parameters can be observed.

At this stage, in both the library and generated files, |inferencemap| distinguishes between several entities.

	Tesselation and point-cell association
		The tesselation that can be viewed as a graph of cells such that an edge is indicative of a neighborhood relationship between the connected cells. A tesselation may be grown from the data but is formely represented as a separate entity.
		However in practical terms the tesselation is conveniently stored together with the complete dataset (or a link to it), as well as the partition of the dataset abstracted as a point-cell association.
		These data are maintained together in such a collection so that the tesselation can be edited with some features that may not necessary while - for example - performing the inference.
		See also the |imt| file format and the :class:`~inferencemap.tesselation.CellStats` datatype.

	Dynamic parameters map
		The dataset is prepared again for the inference so that each cell is reified and can contain both the corresponding locations/translocations and physical parameters.
		The graphical structure of the tesselation is also reflected at this stage in a simplified form so that implementing the inference is made easier.
		See also the |map| file format and the :class:`~inferencemap.inference.Distributed`.


.. |inferencemap| replace:: **InferenceMAP**
.. |imt| replace:: *.imt.h5*
.. |map| replace:: *.map.h5*

