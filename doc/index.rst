.. TRamWAy documentation master file, created by
   sphinx-quickstart on Tue Jan 24 11:46:04 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TRamWAy documentation
=====================

|tramway| helps analyzing single molecule dynamics. It infers the diffusivity, drift, force and potential across space and time.

|tramway| as for its stable version by now is based on Bayesian analysis of Random Walks. In addition to estimating spatial maps of physical parameters, it can generate large amounts of trajectories in the inferred maps using the master equation approximation of the Fokker-Planck describing the motion.

Finally, |tramway| provides multiple representations of both raw data and inferred fields.


Where to start
--------------

.. * :ref:`Installation <installation>`
.. * :ref:`Quick-start guide <quickstart>`
.. * :ref:`Reference library <api>`

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   api

.. More (older) content
 --------------------

.. .. toctree::
    :maxdepth: 1

..    introduction
   installation
   commandline
   fileformat
   api


.. Indices and tables
.. ------------------

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

.. in quickstart.fileformats, 
.. |txt| replace:: *.txt*
.. in quickstart.commandline, quickstart.fileformats, 
.. |xyt| replace:: *.xyt*
.. |trxyt| replace:: *.trxyt*
.. in quickstart, quickstart.helpers, 
.. |h5| replace:: *.h5*
.. in datamodel, 
.. |rwa| replace:: *.rwa*
.. in quickstart.fileformats, 
.. |seconds| replace:: **seconds**
.. |um| replace:: **µm**
.. in index, installation, quickstart, quickstart.concepts, quickstart.fileformats, api, 
.. |tramway| replace:: **TRamWAy**

