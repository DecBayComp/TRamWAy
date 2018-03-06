.. _concepts:

Concepts
========

|tramway| generates maps of the dynamic parameters that dictate the motion of single molecules. 
These parameters may include diffusivity, forces (directional biases), interaction (potential) energies and drift.

|tramway| takes molecule locations or trajectories as input data.

.. image:: concepts.*

A preliminary processing step before generating maps consists of segmentating the space and time into cells that accomodate enough molecule locations so that the inference can generate reliable estimates and, on the other hand, are small enough so that relevant spatial(-temporal) variations in the dynamic parameters can be observed.

This step is referred to as "`tessellation <commandline.html#tessellation>`_" but may consist of temporal windowing for example.

The central and often most time-consuming step consists of `inferring <commandline.html#inference>`_ the value of the parameters in each cell.

As such, maps of these parameters can readily exhibit descriptive information.
A further step consists of extracting features from these maps.


.. |tramway| replace:: **TRamWAy**

