.. _inference:

Inference
=========

This step consists of inferring the values of parameters of the local dynamics in each cell.
These parameters are diffusivity, force, diffusive drift and potential energy.

The inference usually consists of minimizing a cost function.
It can be perfomed in each cell independently (e.g. *D* and *DF*) or jointly in all or some cells (e.g. *DD* and *DV*).

Because the parameters are estimated for each cell, the inference step results in quantitative maps.


Basic usage
-----------

The *tramway* command features the :ref:`infer <commandline_inference>` sub-command that handles this step.
The available inference modes are :ref:`D <inference_d>`, :ref:`DD <inference_dd>`, :ref:`DF <inference_df>` and :ref:`DV <inference_dv>` and can be applied with the :func:`~tramway.helper.inference.infer` helper function:

.. code-block:: python

	from tramway.helper import *

	maps = infer(cells, 'DV')


However such a straightforward call to :func:`~tramway.helper.inference.infer` may occasionally fail.

An argument that should be first considered in an attempt to deal with runtime errors is `localization_error`.
The inference may indeed hit an error because of this value being too small.

Maps for 2D (trans-)location data can be rendered with the :func:`~tramway.helper.inference.map_plot` helper function.
The following command from the :ref:`command-line section <commandline_inference>`::

	> tramway draw map -i example.rwa -L kmeans,df-map0 -cm jet -P size=1,color='w',alpha=.05

can be implemented as follows:

.. code-block:: python

	map_plot('example.rwa', label=('kmeans', 'df-map0'), colormap='jet',
		point_style=dict(size=1, color='w', alpha=.05))



Concepts
--------

Inference
^^^^^^^^^

|tramway| uses the Bayesian inference technique that was first described in [Masson09]_ and implemented in `InferenceMAP <https://research.pasteur.fr/en/software/inferencemap/>`_. 

The motion of single particles is modeled with an overdamped Langevin equation:

.. math::

	\frac{d\textbf{r}}{dt} = \frac{\textbf{F}(\textbf{r})}{\gamma(\textbf{r})} + \sqrt{2D(\textbf{r})} \xi(t)

with :math:`\textbf{r}` the particle location, 
:math:`\textbf{F}(\textbf{r})` the local force (or directional bias), 
:math:`\gamma(\textbf{r})` the local friction coefficient or viscosity, 
:math:`D` the local diffusion coefficient and 
:math:`\xi(t)` a Gaussian noise term.

The model may assume a few additional relationships, 
namely :math:`D(\textbf{r}) \propto \frac{1}{\gamma(\textbf{r})}` 
and :math:`\textbf{F}(\textbf{r}) = - \nabla V(\textbf{r})` 
with :math:`V(\textbf{r})` the local potential energy.

The associated Fokker-Planck equation, which governs the temporal evolution of the particle transition probability :math:`P(\textbf{r}_2, t_2 | \textbf{r}_1, t_1)` is given by:

.. math::

	\frac{dP(\textbf{r}_2, t_2 | \textbf{r}_1, t_1)}{dt} = - \nabla\cdot\left(-\frac{\nabla V(\textbf{r}_1)}{\gamma(\textbf{r}_1)} P(\textbf{r}_2, t_2 | \textbf{r}_1, t_1) - \nabla (D(\textbf{r}_1) P(\textbf{r}_2, t_2 | \textbf{r}_1, t_1))\right)

There is no general analytic solution to the above equation for arbitrary diffusion coefficient :math:`D` and potential energy :math:`V`.
However if we consider a small enough space cell over a short enough time segment, we may assume constant :math:`D` and :math:`V` in each cell, 
upon which the general solution to that equation is a Gaussian distribution described in:

.. math::

	P((\textbf{r}_2, t_2 | \textbf{r}_1, t_1) | D_i, V_i) = \frac{\textrm{exp} - \left(\frac{\left(\textbf{r}_2 - \textbf{r}_1 + \frac{\nabla V_i (t_2 - t_1)}{\gamma_i}\right)^2}{4 \left(D_i + \frac{\sigma^2}{t_2 - t_1}\right)(t_2 - t_1)}\right)}{4 \pi \left(D_i + \frac{\sigma^2}{t_2 - t_1}\right)(t_2 - t_1)}

with :math:`i` the index for the cell, :math:`(\textbf{r}_1, t_1)` and :math:`(\textbf{r}_2, t_2)` two points in cell :math:`i` and :math:`\sigma` the experimental localization error.

The probability of local parameters :math:`D_i` and :math:`V_i` is calculated from the set of local translocations :math:`T_i=\{( \Delta\textbf{r}_j, \Delta t_j )\}_j` applying Bayes' rule:

.. math::

	P( D_i, V_i | T_i ) = \frac{P( T_i | D_i, V_i ) P( D_i, V_i )}{P(T_i)}

and assuming:

.. math::

	P( T_i | D_i, V_i ) = \prod_j P( \Delta\textbf{r}_j, \Delta t_j | D_i, V_i )

:math:`P(D,V|T)` is the *posterior probability*, :math:`P(D,V)` is the *prior probability* and :math:`P(T)` is the evidence which is treated as a normalization constant.

For each cell, :math:`P(D,V|T)` is optimized for the model parameters :math:`D` and :math:`V` or :math:`\textbf{F}`.


.. [Masson09] Masson J.-B., Casanova D., Türkcan S., Voisinne G., Popoff M.R., Vergassola M. and Alexandrou A. (2009) Inferring maps of forces inside cell membrane microdomains, *Physical Review Letters* 102(4):048103


Maps
^^^^

The raw inferred maps are usually of type pandas' `DataFrame` with column names such as *diffusivity*, *potential*, *force x*, *force y* where *x* and *y* refers to space dimensions.

The maps are encapsulated in :class:`~tramway.inference.base.Maps` objects that are transitional constructs to handle former formats for maps.
The :class:`~tramway.inference.base.Maps` class is also a convenient container to store information about the method and parameters used to generate the encapsulated maps (see attribute :attr:`~tramway.inference.base.Maps.maps`).


Distributed cells
^^^^^^^^^^^^^^^^^

The :func:`~tramway.helper.inference.infer` function prepares the :class:`~tramway.tessellation.base.CellStats` partition (see the :ref:`tessellation` section) before the inference is run.

Cells are represented by either :class:`~tramway.inference.base.Locations` or :class:`~tramway.inference.base.Translocations` objects. 
Both types of objects derivate from the :class:`~tramway.inference.base.Cell` class.

These cell objects are distributed in a :class:`~tramway.inference.base.Distributed` object.
The :class:`~tramway.inference.base.Distributed` class controls how the cells and the associated (trans-)locations are passed to the inference algorithm.

For example cells can be grouped in subsets of cells.
In this case the top :class:`~tramway.inference.base.Distributed` object will contain other :class:`~tramway.inference.base.Distributed` objects that will in turn contain :class:`~tramway.inference.base.Cell` objects.

The main routine of an inference plugin receives a :class:`~tramway.inference.base.Distributed` object and can:

* iterate over the contained cells (:class:`~tramway.inference.base.Distributed` features a dict-like interface),
* take benefit from the cell adjacency matrix (attribute :attr:`~tramway.inference.base.Distributed.adjacency`)
* and other convenience calculations such as gradient components (method :meth:`~tramway.inference.base.Distributed.grad`) that can be summed (method :meth:`~tramway.inference.base.Distributed.grad_sum`).


The :meth:`~tramway.inference.base.Distributed.run` applies the inference routine on the defined subsets of cells.
It handles the multi-processing logic and combines the regional maps into a full map.
The number of workers (or processes) can be set with the `worker_count` argument.


Methods
-------

Inference modes are made available as plugins.
Some of them are listed below:


.. list-table:: Available inference modes
   :header-rows: 1

   * - Inference mode
     - Parameters
     - Speed
     - Generated maps

   * - :ref:`D <inference_d>`
     - | :math:`D`
     - fast
     - | diffusivity

   * - :ref:`DF <inference_df>`
     - | :math:`D`
       | :math:`\textbf{F}`
     - fast
     - | diffusivity
       | force

   * - :ref:`DD <inference_dd>`
     - | :math:`D`
       | :math:`\frac{\textbf{F}}{\gamma}` [#a]_
     - medium
     - | diffusivity
       | drift [#b]_

   * - :ref:`DV <inference_dv>`
     - | :math:`D`
       | :math:`V`
       | :math:`\textbf{F}`
     - slow
     - | diffusivity
       | potential
       | force [#c]_


.. [#a] :math:`\frac{\textbf{F}}{\gamma}` is approximated as :math:`D\nabla D`
.. [#b] not a direct product of optimizing; derived from the diffusivity
.. [#c] not a direct product of optimizing; derived from the potential energy


All the methods use :math:`\sigma = 30 \textrm{nm}` as default value for the experimental localization error.

They also feature an optional Jeffreys' prior that may be introduced in the posterior probability with the ``-j`` command-line option or the `jeffreys_prior` argument to :func:`~tramway.helper.inference.infer`.
In the expressions below, it is referred to as :math:`P_J(D_i)`.

Most 

.. _inference_d:

*D* inference
^^^^^^^^^^^^^

This inference mode estimates solely the diffusion coefficient in each cell independently, resulting in a rapid computation.
The posterior probability used to infer the diffusivity :math:`D_i` in cell :math:`i` given the corresponding set of translocations :math:`T_i = {(\Delta\textbf{r}_j, \Delta t_j)}_j` is given by:

.. math::

	P(D_i | T_i) \propto \prod_j \frac{\textrm{exp}\left(-\frac{\Delta\textbf{r}_j^2}{4\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}\right)}{4\pi\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}P_J(D_i)

The *D* inference mode is well-suited to freely diffusing molecules and the rapid characterization of the diffusivity.


.. _inference_df:

*DF* inference
^^^^^^^^^^^^^^

::

	TODO 


.. math::

	P(D_i, \textbf{F}_i | T_i) \propto \prod_j \frac{\textrm{exp}\left(-\frac{\left(\Delta\textbf{r}_j - \frac{D_i\textbf{F}_i\Delta t_j}{k_BT}\right)^2}{4\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}\right)}{4\pi\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}P_J(D_i)


.. _inference_dd:

*DD* inference
^^^^^^^^^^^^^^

*DD* stands for *Diffusivity and Drift*.
This mode is very similar to the :ref:`D mode <inference_d>`.
It adds a smoothing factor :math:`P_S(\textbf{D},i)` that penalizes the local diffusivity gradient at cell :math:`i`. 

This factor acts like a prior probability. 
However the notation :math:`P_S(\textbf{D},i)` here is not that of a joint probability but that of a function instead.
Indeed, :math:`P_S` is a function of the diffusivity at all the cells - denoted here by vector :math:`\textbf{D}` - and evaluated at cell :math:`i`.

As a consequence, the posterior probability is jointly optimized at all the cells, 
which supposes a higher computational cost.
On the other side, penalizing the local gradients helps to get diffusivity landscapes such that :math:`\Delta D` is well-behaved.
Indeed, :math:`\Delta D` may hopefully not exhibit extreme values that would make no physical sense.

:math:`D\Delta D` is used as an approximation of the drift :math:`\frac{\textbf{F}}{\gamma}`.

The maximized posterior probability is given by:

.. math::

		P(D_i | T_i) \propto \prod_j \frac{\textrm{exp}\left(-\frac{\Delta\textbf{r}_j^2}{4\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}\right)}{4\pi\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}P_J(D_i)P_S(\textbf{D},i)

The *DD* inference mode is well-suited to active processes (e.g. active transport phenomena).


.. _inference_dv:

*DV* inference
^^^^^^^^^^^^^^

::

	TODO 


.. math::

	P(D_i, V_i | T_i) \propto \prod_j \frac{\textrm{exp}\left(-\frac{\left(\Delta\textbf{r}_j + \frac{D_i\nabla V_i\Delta t_j}{k_BT}\right)^2}{4\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}\right)}{4\pi\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}P_J(D_i)P_S(\textbf{D},i)P_S(\textbf{V},i)



Advanced usage
--------------

Fuzzy cell-point association
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Custom gradient
^^^^^^^^^^^^^^^



.. |tramway| replace:: **TRamWAy**

