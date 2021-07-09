.. _inference:

Inference
=========

This step consists of inferring the values of parameters of the local dynamics in each microdomain or cell.
These parameters can be diffusivities, forces, drifts and potential energies.

The inference usually consists of minimizing a cost function.
It can be perfomed in each cell independently (e.g. :ref:`D <inference_d>`, :ref:`DD <inference_dd>` and :ref:`DF <inference_df>` in their *degraded* variants) or jointly in all or some cells (e.g. :ref:`DV <inference_dv>`).

Because the parameters are estimated for each cell, the resulting parameter values can be rendered as quantitative maps.


Basic usage
-----------

The *tramway* command features the :ref:`infer <commandline_inference>` sub-command that handles this step.
The available inference modes are :ref:`D <inference_d>`, :ref:`DD <inference_dd>`, :ref:`DF <inference_df>` and :ref:`DV <inference_dv>` 
(also referred to as :ref:`(D) <inference_d>`, :ref:`(D,Drift) <inference_dd>`, :ref:`(D,F) <inference_df>` and :ref:`(D,V) <inference_dv>` respectively in `InferenceMAP`_) 
and can be applied with the :func:`~tramway.helper.inference.infer` helper function:

.. code-block:: python

	from tramway.helper import infer

	maps = infer(cells, 'DV')


However such a straightforward call to :func:`~tramway.helper.inference.infer` may occasionally fail in situations where the observed molecules exhibit little movement, that cannot even account for the experimental localisation precision.

An argument that should be first considered in an attempt to deal with runtime errors is `min_diffusivity`.

Another important argument that can help is the localisation precision or error (`sigma` or `sigma2`).
See also the `Common parameters and default values`_ section.


Maps for 2D (trans-)location data can be rendered with the :func:`~tramway.helper.inference.map_plot` helper function.
The following command from the :ref:`command-line section <commandline_inference>`::

	> tramway draw map -i example.rwa -L kmeans,df-map0 -cm jet -P size=1,color='w',alpha=.05

can be implemented as follows:

.. code-block:: python

	map_plot('example.rwa', label=('kmeans', 'df-map0'), colormap='jet',
		point_style=dict(size=1, color='w', alpha=.05))



Concepts
--------

|tramway| uses the Bayesian inference technique that was first described in [Masson09]_ and implemented in `InferenceMAP`_. 

The motion of single particles is modeled with an overdamped Langevin equation:

.. math::

	\frac{d\textbf{r}}{dt} = \frac{\textbf{F}(\textbf{r})}{\gamma(\textbf{r})} + \sqrt{2D(\textbf{r})} \xi(t)

with :math:`\textbf{r}` the particle location, 
:math:`\textbf{F}(\textbf{r})` the local force (or directional bias), 
:math:`\gamma(\textbf{r})` the local friction, proportional to the viscosity, 
:math:`D` the local diffusion coefficient and 
:math:`\xi(t)` a Gaussian noise term.

The :ref:`DV <inference_dv>` model additionally assumes :math:`\textbf{F}(\textbf{r}) = -\nabla V(\textbf{r})` 
with :math:`V(\textbf{r})` the local potential energy.

The associated Fokker-Planck equation, which governs the temporal evolution of the particle transition probability :math:`P(\textbf{r}_2, t_2 | \textbf{r}_1, t_1)` is given by:

.. math::

	\frac{dP(\textbf{r}_2, t_2 | \textbf{r}_1, t_1)}{dt} = - \nabla\cdot\left(-\frac{\nabla V(\textbf{r}_1)}{\gamma(\textbf{r}_1)} P(\textbf{r}_2, t_2 | \textbf{r}_1, t_1) - \nabla (D(\textbf{r}_1) P(\textbf{r}_2, t_2 | \textbf{r}_1, t_1))\right)

There is no general analytic solution to the above equation for arbitrary diffusion coefficient :math:`D` and potential energy :math:`V`.
However if we consider a small enough space cell over a short enough time segment, we may assume constant :math:`D` and :math:`V` in each cell, 
upon which the general solution to that equation leads to the following likelihood:

.. math::

	P((\textbf{r}_2, t_2 | \textbf{r}_1, t_1) | D_i, V_i) = \frac{\textrm{exp} \left(- \frac{\left(\textbf{r}_2 - \textbf{r}_1 + \frac{\nabla V_i (t_2 - t_1)}{\gamma_i}\right)^2}{4 \left(D_i + \frac{\sigma^2}{t_2 - t_1}\right)(t_2 - t_1)}\right)}{4 \pi \left(D_i + \frac{\sigma^2}{t_2 - t_1}\right)(t_2 - t_1)}

with :math:`i` the index for the cell, :math:`(\textbf{r}_1, t_1)` and :math:`(\textbf{r}_2, t_2)` two points in cell :math:`i` and :math:`\sigma` the experimental localization error.

The probability of the local parameters :math:`D_i` and :math:`V_i` is calculated from the set of local translocations :math:`T_i=\{( \Delta\textbf{r}_j, \Delta t_j )\}_j` applying Bayes' rule:

.. math::

	P( D, V | T ) = \frac{P( T | D, V ) P( D, V )}{P(T)}

and, introducing the mapping hypothesis to decompose the likelihood:

.. math::

	P( T | D, V ) = \prod_i P( T_i | D_i, V_i ) = \prod_i \prod_j P( \Delta\textbf{r}_j, \Delta t_j | D_i, V_i )

:math:`P(D,V|T)` is the *posterior probability*, :math:`P(D,V)` is the *prior probability* and :math:`P(T)` is the *evidence*, that can be ignored when maximizing the posterior.

Models other than :ref:`DV <inference_dv>` follow the same rule, with :math:`V` substituted by other model parameters.

.. [Masson09] Masson J.-B., Casanova D., Türkcan S., Voisinne G., Popoff M.R., Vergassola M. and Alexandrou A. (2009) Inferring maps of forces inside cell membrane microdomains, *Physical Review Letters* 102(4):048103


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

   * - :ref:`DD <inference_dd>`
     - | :math:`D`
       | :math:`\frac{\textbf{F}}{\gamma}`
     - fast
     - | diffusivity
       | drift

   * - :ref:`DF <inference_df>`
     - | :math:`D`
       | :math:`\textbf{F}` [#a]_
     - fast
     - | diffusivity
       | force

   * - :ref:`DV <inference_dv>`
     - | :math:`D`
       | :math:`V` [#a]_
       | :math:`\textbf{F}` [#a]_
     - slow
     - | diffusivity
       | potential
       | force [#b]_


.. [#a] the potentials and forces are estimated as :math:`\frac{V}{k_{\textrm{B}}T}` and :math:`\frac{\textbf{F}}{k_{\textrm{B}}T}` respectively
.. [#b] not a direct product of optimizing; derived from the potential energy


.. _inference_d:

*D* inference
^^^^^^^^^^^^^

This inference mode estimates solely the diffusion coefficient in each cell independently, resulting in a rapid computation.
The likelihood used to infer the local diffusivity :math:`D_i` in cell :math:`i` given the corresponding set of translocations :math:`T_i = {(\Delta\textbf{r}_j, \Delta t_j)}_j` is given by:

.. math::

	P(T_i | D_i) \propto \prod_j \frac{\textrm{exp}\left(-\frac{\Delta\textbf{r}_j^2}{4\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}\right)}{4\pi\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}

The *D* inference mode is well-suited to freely diffusing molecules and the rapid characterization of the diffusivity.

This mode supports the :ref:`Jeffreys' prior <inference_jeffreys>` and the :ref:`diffusivity smoothing prior <inference_smoothing>`.

.. _inference_dd:

*DD* inference
^^^^^^^^^^^^^^

*DD* stands for *Diffusivity and Drift*.

This mode is very similar to the :ref:`DF mode <inference_df>` mode. 
The whole drift :math:`\textbf{a} = \frac{\textbf{F}}{\gamma}` is optimized instead of the force :math:`\textbf{F}`. 
This may offer increased stability in the optimization. 
Indeed the contribution of the drift to the objective function does not depend directly on the simultaneously estimated diffusivity.

The likelihood is given by:

.. math::

	P(T_i | D_i, \textbf{a}_i) \propto \prod_j \frac{\textrm{exp}\left(-\frac{\left(\Delta\textbf{r}_j - \textbf{a}_i\Delta t_j\right)^2}{4\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}\right)}{4\pi\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}

If space (:math:`\textbf{r}`) is measured as :math:`\mu m`, the unit for the drift magnitude is :math:`\mu m s^{-1}`.

The *DD* inference mode is well-suited to active processes (e.g. active transport phenomena).

This mode supports the :ref:`Jeffreys' prior <inference_jeffreys>`, and the :ref:`diffusivity smoothing prior <inference_smoothing>`.

.. _inference_df:

*DF* inference
^^^^^^^^^^^^^^

This inference mode estimates the diffusivity and force.
It takes advantage of the assumption that :math:`D(\textbf{r}) = \frac{k_{\textrm{B}}T}{\gamma(\textbf{r})}`.

The likelihood used to infer the local diffusivity :math:`D_i` and force :math:`\textbf{F}_i` is given by:

.. math::

	P(T_i | D_i, \textbf{F}_i) \propto \prod_j \frac{\textrm{exp}\left(-\frac{\left(\Delta\textbf{r}_j - D_i\frac{\textbf{F}_i}{k_{\textrm{B}}T}\Delta t_j\right)^2}{4\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}\right)}{4\pi\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}

The *DF* inference mode is well-suited to mapping local force components, especially in the presence of non-potential forces (e.g. a rotational component).
This mode allows for the rapid characterization of the diffusivity and directional biases of the trajectories.

This mode supports the :ref:`Jeffreys' prior <inference_jeffreys>` and the :ref:`diffusivity smoothing prior <inference_smoothing>`.

Following `InferenceMAP`_, TRamWAy estimates the scaled force :math:`\frac{\textbf{F}}{k_{\textrm{B}}T}`.
As a consequence, force components and magnitude can be expressed as :math:`k_{\textrm{B}}T\mu m^{-1}`, using :math:`k_{\textrm{B}}T` as a unit for energy, and :math:`\mu m` the unit for space (in the case :math:`\textbf{r}` is expressed as :math:`\mu m`).

Note anyway that forces are often best displayed as logarithms.

.. _inference_dv:

*DV* inference
^^^^^^^^^^^^^^

Building up on the *DF* model, this model introduces an additional assumption on the distribution of the directional biases, and considers conservative forces only :math:`\textbf{F}=-\nabla V`, with :math:`V` the effective potential.

The likelihood becomes:

.. math::

	P(T_i | D_i, V_i) \propto \prod_j \frac{\textrm{exp}\left(-\frac{\left(\Delta\textbf{r}_j + D_i\frac{\nabla V_i}{k_{\textrm{B}}T}\Delta t_j\right)^2}{4\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}\right)}{4\pi\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}

Following `InferenceMAP`_, TRamWAy estimates the dimension-less ratio :math:`\frac{V}{k_{\textrm{B}}T}`.
As such, :math:`V` can be expressed as :math:`k_{\textrm{B}}T`.

Because this model requires access to the neighbour cells/bins for estimating the local potential gradient :math:`\nabla V_i`,
the overall posterior probability is maximized necessarily optimizing all the spatially distributed parameters simultaneously.

As a consequence, this method is slow but smoothing priors can be introduced at little extra computational cost.
The smoothing factors are described in a :ref:`dedicated section <inference_smoothing>`.

This mode also supports the :ref:`Jeffreys' prior <inference_jeffreys>`.

More information can also be found about :ref:`gradient calculation <gradient>`.

Stochastic DV
"""""""""""""

A key variant of the default *DV* mode is the *stochastic.dv* mode, which randomly picks and chooses a cell at each iteration and performs a gradient descent step on the associated parameters considering the neighbour cells instead of the full tessellation.

It is showcased in most of the tutorial notebooks, *e.g.* `basic/inference.ipynb <https://github.com/DecBayComp/TRamWAy/blob/master/notebooks/basic/inference.ipynb>`_ and `RWAnalyzer tour.ipynb <https://github.com/DecBayComp/TRamWAy/blob/master/notebooks/RWAnalyzer%20tour.ipynb>`_, and is especially useful for inferring dynamic maps with temporal smoothing.


.. _inference_parameters:

Common parameters and default values
------------------------------------

All the methods use :math:`\sigma = 0.03 \textrm{µm}` as default value for the experimental localization error.
This parameter is defined by the experimental setup and can be set in |tramway| with the ``--sigma`` command-line option or the `sigma` argument to :func:`~tramway.helper.inference.infer` and is expressed in |um|.

Compare::

	> tramway -i example.rwa infer dd --sigma 0.01 -l DD_sigma_10nm

.. code-block:: python

	from tramway.helper import infer

	infer('example.rwa', 'dd', sigma=0.01, output_label='DD_sigma_10nm')

If no specific prior is defined, a uniform prior is used by default.

.. _inference_jeffreys:

Jeffreys' prior
^^^^^^^^^^^^^^^

All the methods described here also feature an optional Jeffreys' prior on the diffusivity. 
It is a non-informative prior used to make the posterior probability distribution less sensitive to re-parametrization of diffusivity :math:`D`.

This prior - referred to as :math:`P_J(D_i)` - modifies the maximized posterior probability:

.. math::

	P^*(D_i, ... | T_i) \propto P(T_i | D_i, ...) P_J(D_i)

Its value varies depending on the inference mode. Compare:

.. list-table:: Jeffreys' prior for the different inference modes
   :header-rows: 1

   * - Inference mode
     - Jeffreys' prior :math:`P_J(D_i)`

   * - :ref:`D <inference_d>`
     - :math:`\frac{1}{\left(D_i\overline{\Delta t}_i + \sigma^2\right)^2}`

   * - :ref:`DD <inference_dd>`
     - :math:`\frac{1}{\left(D_i\overline{\Delta t}_i + \sigma^2\right)^2}`

   * - :ref:`DF <inference_df>`
     - :math:`\frac{D_i^2}{\left(D_i\overline{\Delta t}_i + \sigma^2\right)^2}`

   * - :ref:`DV <inference_dv>`
     - :math:`\frac{D_i^2}{\left(D_i\overline{\Delta t}_i + \sigma^2\right)^2}`


The Jeffreys' prior may be introduced in the posterior probability with the ``-j`` command-line option or the `jeffreys_prior` argument to :func:`~tramway.helper.inference.infer`.
Compare::

	> tramway -i example.rwa infer dd -j -l DD_jeffreys

.. code-block:: python

	from tramway.helper import infer

	infer('example.rwa', 'dd', jeffreys_prior=True, output_label='DD_jeffreys')


Note that with this prior the default minimum diffusivity value is :math:`0.01`. 
Consider modifying this value.


.. _inference_smoothing:

Spatial smoothing priors
^^^^^^^^^^^^^^^^^^^^^^^^

A smoothing (improper) prior penalizes the gradients or spatial variations of the inferred parameters. 
It is meant to reinforce the physical plausibility of the inferred maps. 
For example, in certain situations we do not expect large changes in the diffusion coefficient between neighbour cells.

An optional smoothing factor, for example :math:`P_S(\textbf{D})` for the diffusivity, multiplies with the original expression of the posterior probability and penalizes all the diffusivity gradients. 
:math:`P_S` is a function of the diffusivity at all the cells, hence the vectorial notation :math:`\textbf{D}` for the diffusivity.

The maximized probability becomes:

.. math::

	P^*(\textbf{D}, ... | T) = P_S(\textbf{D}) \prod_i P(T_i | D_i, ...)

with, for example:

.. math::

	P_S(\textbf{D}) = \textrm{exp}\left(-\mu\sum_i \mathcal{A}_i||\nabla D_i||^2\right)

where :math:`\mathcal{A}_i` is the area of bin :math:`i`.

The :math:`\mu` parameter can be set with the ``-d`` command-line option or the `diffusivity_prior` argument to :func:`~tramway.helper.inference.infer`.
Compare::

	> tramway -i example.rwa infer dd -d 1 -l DD_d_1

.. code-block:: python

	from tramway.helper import infer

	infer('example.rwa', 'dd', diffusivity_prior=1., output_label='DD_d_1')



Note that the :ref:`DV <inference_dv>` inference mode readily features this smoothing factor, in addition to a similar smoothing factor :math:`P_S(\textbf{V})` for the potential energy:

.. math::

	P_S(\textbf{V}) = \textrm{exp}\left(-\lambda\sum_i \mathcal{A}_i||\nabla V_i||^2\right)

Similarly to :math:`\mu`, the :math:`\lambda` parameter can be set with the ``-v`` command-line option or the `potential_prior` argument to :func:`~tramway.helper.inference.infer`.

More information can be found about :ref:`gradient calculation <gradient>`.

Alternative penalties
"""""""""""""""""""""

Gradients :math:`\nabla X` are tangents and may not catch all the spatial variations,
especially in the case of a regular mesh with an oscillating :math:`X`.

From version *0.4*, all the methods feature the ``rgrad='delta'`` argument
that replaces :math:`\nabla X_i` in :math:`P_S(\textbf{X})` by :math:`\Delta X_i`
as described in :func:`~tramway.inference.gradient.delta0`
that considers the actual differences in :math:`X` with the neighbour bins.

Beware that, in future versions, this alternative penalty may become the default behaviour.
To keep these methods penalize the gradient, set ``rgrad='grad'``.

Temporal smoothing prior
^^^^^^^^^^^^^^^^^^^^^^^^

In combination with a time window, the dynamic maps can be inferred considering parameter smoothing across time.

Temporal smoothing is available in the *stochastic.dv* mode.
It is implemented with yet another improper prior:

.. math::

    P''_S(\textbf{D},\textbf{V}) = \textrm{exp}\left(-\left(\tau_\mu\sum_i \frac{\partial D_i}{\partial t}^2 + \tau_\lambda\sum_i \frac{\partial V_i}{\partial t}^2\right)\right)

:math:`\tau_\mu` and :math:`\tau_\lambda` are available as arguments *diffusivity_time_prior*/*diffusion_time_prior* and *potential_time_prior* respectively.

See also the following `notebook <https://github.com/DecBayComp/COMPARE19/blob/master/1.%20Simulated%20data/2.%20dynamic%20inference.ipynb>`_.


Implementation details
----------------------

Maps
^^^^

The maps are available as :class:`~tramway.inference.base.Maps` objects that expose a `pandas.DataFrame`-like interface with "column" names such as '*diffusivity*', '*potential*' and '*force*'.

``maps['force']`` for 2D space-only data will typically return a :class:`~pandas.DataFrame` with two columns '*force x*' and '*force y*', where *x* and *y* refers to the space dimensions.


Distributed cells
^^^^^^^^^^^^^^^^^

The :func:`~tramway.helper.inference.infer` function prepares the :class:`~tramway.tessellation.base.Partition` (see the :ref:`tessellation` section) before the inference is run.

Cells are represented by either :class:`~tramway.inference.base.Locations` or :class:`~tramway.inference.base.Translocations` objects. 
Both types of objects derivate from the :class:`~tramway.inference.base.Cell`/:class:`~tramway.inference.base.FiniteElement` class.

These cell objects are grouped together in a dict-like :class:`~tramway.inference.base.FiniteElements` object.
The :class:`~tramway.inference.base.FiniteElements` class controls how the cells and the associated (trans-)locations are passed to the inference algorithm.

For example cells can be grouped in subsets of cells.
In this case the top :class:`~tramway.inference.base.FiniteElements` object will contain other :class:`~tramway.inference.base.FiniteElements` objects that will in turn contain :class:`~tramway.inference.base.Cell` objects.

The main routine of an inference plugin receives a :class:`~tramway.inference.base.FiniteElements` object and can:

* iterate over the contained cells (:class:`~tramway.inference.base.FiniteElements` features a dict-like interface),
* take benefit from the cell adjacency matrix (attribute :attr:`~tramway.inference.base.FiniteElements.adjacency`)
* and other convenience calculations such as gradient components (method :meth:`~tramway.inference.base.FiniteElements.grad`) that can be summed (method :meth:`~tramway.inference.base.FiniteElements.grad_sum`).


The :meth:`~tramway.inference.base.FiniteElements.run` applies the inference routine on the defined subsets of cells.
It handles the multi-processing logic and combines the regional maps into a full map.
The number of workers (or processes) can be set with the `worker_count` argument.


.. _inference_bayes_factor:

Force testing
-------------

In every cell, the inferred drift can be compared against the effect of diffusivity gradients.

The `bayes_factor`_ module calculates the odds (the probability ratio) of having an actual active force
over the probability that diffusivity gradients can explain the observed drift.
The user-specified `B_threshold` threshold sets the required level of evidence.
Values above `B_threshold` indicate the presence of an active force,
and values below `1/B_threshold` indicate that diffusivity gradients are the moste likely explanation of the observed drift.
The values in-between indicate that a conclusion cannot be reached at the required level of evidence.

The `bayes_factor`_ plugin generates 3 additional maps:

* `lg_B`: current Bayes factor value
* `force`: ternary map for the presence of an active force (``-1``: no force, ``0``: insufficient evidence, ``1``: force)
* `min_n`: given the supplied total force and diffusivity gradient estimates are correct, returns a number of points to be collected in the current bin, so as to reach the required level of evidence.

The `bayes_factor`_ plugin operates on top of a diffusivity map that must be inferred first, preferably with the *d.conj_prior* plugin.

The current version of the `bayes_factor`_ plugin does not test the drift or force inferred by plugins such as :ref:`DD <inference_dd>`, :ref:`DF <inference_df>` or :ref:`DV <inference_dv>`.


.. Advanced usage
.. --------------

.. Fuzzy cell-point association
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. Custom gradient
.. ^^^^^^^^^^^^^^^


