.. _quickstart.concepts:

Concepts
========

.. warning:: This section is outdated and needs to be reworked.

|tramway| generates maps of the dynamic parameters that dictate the motion of single molecules. These parameters may include the diffusion, forces (directional biases), interaction (potential) energies and drift.

|tramway| takes molecule locations or trajectories data as input.

A preliminary processing step before generating maps consists of segmentating space and time into cells that accomodate enough molecule locations so that the inference is made possible and, on the hand, are small enough so that relevant spatial(-temporal) variations in the dynamic parameters can be observed.

At this stage, |tramway| distinguishes between several entities.

	Tessellation and point-cell association
		The tessellation that can be viewed as a graph of cells such that an edge is indicative of a neighborhood relationship between the connected cells. 
		A tessellation may be grown from the data and is formaly represented as a separate entity.
		These data are maintained together in such a collection so that the tessellation can be edited with some features that may not necessary while - for example - performing the inference.
		See also the :class:`~tramway.tessellation.CellStats` datatype.

	Dynamic parameters map
		The dataset is prepared again for the inference so that each cell is reified and can contain both the corresponding locations/translocations and physical parameters.
		The graphical structure of the tessellation is also reflected at this stage in a simplified form so that implementing the inference is made easier.
		See also the :class:`~tramway.inference.Distributed`.


.. |tramway| replace:: **TRamWAy**
.. |rwa| replace:: *.rwa*

