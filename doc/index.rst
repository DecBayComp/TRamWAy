
=======
TRamWAy
=======

|tramway| is a tool to analyse single molecule dynamics using their locations or trajectories.
Parameters of single molecule dynamics such as diffusivity, drift, force and potential energy are resolved in space and time.

|tramway| has been designed as a modular Python library that may accomodate additional plugins to sample the localization microscopy data and infer model parameters in microdomains.

The full code is distributed under :ref:`CeCILL license <license>` and is available at `github`_.


--------
Features
--------

.. * :ref:`deconvolution <deconvolution>` of stacks of images

* tracking of localization microscopy data
* spatial :ref:`tessellation <tessellation>` and temporal segmentation
* :ref:`inference <inference>` of diffusivity, drift, force, potential energy, etc.
* analyses of the estimated force, such as curl calculation and :ref:`Bayes factor <inference_bayes_factor>` calculation to distinguish between interactions and spurious forces
* and more: generation of random walk trajectories, plotting utilities, etc



-------------
Quick example
-------------

Maps of diffusivity and effective potential estimates can be generated from molecule trajectories with as few as three commands:

.. code-block:: shell
	:linenos:

	tramway tessellate gwr -i trajectories.txt -o my_analyses.rwa
	tramway infer dv -i my_analyses.rwa
	tramway draw map -i my_analyses.rwa

* Single molecule locations and/or trajectories are usually stored in text files.
  In the above example, such an input file is named *trajectories.txt*.
* The first step (``1``) consists of importing these location data and tessellating the space into cells of adequate size. 
  The *GWR* method grows a graph that fits molecule density.
* All the analysis results derived from a same dataset are stored in a single *.rwa* file.
* The second step (``2``) consists of inferring parameters of the molecule dynamics,
  for example local diffusivities and potential energies (*DV* mode).
  This may take a while and recruit most of the processing units of the host.
* The estimated values can now be visualized as 2D maps (``3``).

.. tabularcolumns:: |p{0.25\linewidth}|p{0.25\linewidth}|p{0.25\linewidth}|p{0.25\linewidth}|

+-------------------------+-------------------------+---------------------------------------------------+-------------------------+
|   molecule locations    |    tessellation (1)     |                                  maps (2,3)                                 |
|                         |                         +-------------------------+-------------------------+-------------------------+
|                         |                         |      diffusivity        |    potential energy     |         force           |
+=========================+=========================+=========================+=========================+=========================+
| .. image:: t0-0.*       | .. image:: t0-1.*       | .. image:: t0-2.*       | .. image:: t0-3.*       | .. image:: t0-4.*       |
+-------------------------+-------------------------+-------------------------+-------------------------+-------------------------+

The equivalent Python code is:

.. code-block:: python

	from tramway.helper import *

	tessellate('trajectories.txt', 'gwr', output_file='my_analyses.rwa')
	infer('my_analyses.rwa', 'dv')
	map_plot('my_analyses.rwa')


--------------
Where to start
--------------

|tramway| is distributed under the :ref:`terms of the CeCILL license <license>` and you should first get to know these terms.

* :ref:`Installation <installation>`
* :ref:`Concepts <concepts>`
* :ref:`Command-line <commandline>`
* :ref:`Tutorials <tutorials>`
* :ref:`Reference library <api>`


.. toctree::
	:caption: Getting started
	:name: getting_started
	:maxdepth: 1
	:hidden:

	installation
	concepts
	commandline
	tutorials
	license

.. toctree::
	:caption: User manual
	:name: usage
	:maxdepth: 1
	:hidden:

	datamodel
	tessellation
	inference

.. toctree::
	:caption: Developping
	:name: advanced
	:maxdepth: 1
	:hidden:

	api
..	plugins


