.. _inference:

Inference
=========

This step consists of inferring the values of parameters of the local dynamics in each cell.
These parameters are diffusivity, force, diffusive drift and potential energy.

The inference usually consists of minimizing a cost function.
It can be perfomed in each cell independently (e.g. :ref:`D <inference_d>`, :ref:`DF <inference_df>` and :ref:`DD <inference_dd>` with no smoothing) or jointly in all or some cells (e.g. :ref:`DV <inference_dv>`).

Because the parameters are estimated for each cell, the resulting parameter values can be rendered as quantitative maps.


Basic usage
-----------

The *tramway* command features the :ref:`infer <commandline_inference>` sub-command that handles this step.
The available inference modes are :ref:`D <inference_d>`, :ref:`DF <inference_df>`, 
:ref:`DD <inference_dd>` and :ref:`DV <inference_dv>` 
(also referred to as :ref:`(D) <inference_d>`, :ref:`(D,F) <inference_df>`, :ref:`(D,Drift) <inference_dd>` and :ref:`(D,V) <inference_dv>` respectively in `InferenceMAP`_) 
and can be applied with the :func:`~tramway.helper.inference.infer` helper function:

.. code-block:: python

	from tramway.helper import infer

	maps = infer(cells, 'DV')


However such a straightforward call to :func:`~tramway.helper.inference.infer` may occasionally fail in situations where the observed diffusivities are low and the experimental localization error is not properly set.

An argument that should be first considered in an attempt to deal with runtime errors is `localization_error` (or equivalently the ``-e`` command-line option).
The inference may indeed hit an error because of this value being too small.
See also the `Priors and default values`_ section.


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

|tramway| uses the Bayesian inference technique that was first described in [Masson09]_ and implemented in `InferenceMAP`_. 

The motion of single particles is modeled with an overdamped Langevin equation:

.. math::

	\frac{d\textbf{r}}{dt} = \frac{\textbf{F}(\textbf{r})}{\gamma(\textbf{r})} + \sqrt{2D(\textbf{r})} \xi(t)

with :math:`\textbf{r}` the particle location, 
:math:`\textbf{F}(\textbf{r})` the local force (or directional bias), 
:math:`\gamma(\textbf{r})` the local friction coefficient or viscosity, 
:math:`D` the local diffusion coefficient and 
:math:`\xi(t)` a Gaussian noise term.

The model additionally assumes :math:`\textbf{F}(\textbf{r}) = - \nabla V(\textbf{r})` 
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

The maps are available as :class:`~tramway.inference.base.Maps` objects that expose a `pandas.DataFrame`-like interface with "column" names such as '*diffusivity*', '*potential*' and '*force*'.

``maps['force']`` for 2D space-only data will typically return a :class:`~pandas.DataFrame` with two columns '*force x*' and '*force y*', where *x* and *y* refers to the space dimensions.


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
       | :math:`\textbf{F}` [#a]_
     - fast
     - | diffusivity
       | force

   * - :ref:`DD <inference_dd>`
     - | :math:`D`
       | :math:`\frac{\textbf{F}}{\gamma}` [#a]_
     - fast
     - | diffusivity
       | drift

   * - :ref:`DV <inference_dv>`
     - | :math:`D`
       | :math:`V` [#a]_
       | :math:`\textbf{F}` [#a]_
     - slow
     - | diffusivity
       | potential
       | force [#b]_


.. [#a] the amplitude of directional biases is expressed in numbers of :math:`k_BT`
.. [#b] not a direct product of optimizing; derived from the potential energy


.. _inference_d:

*D* inference
^^^^^^^^^^^^^

This inference mode estimates solely the diffusion coefficient in each cell independently, resulting in a rapid computation.
The posterior probability used to infer the diffusivity :math:`D_i` in cell :math:`i` given the corresponding set of translocations :math:`T_i = {(\Delta\textbf{r}_j, \Delta t_j)}_j` is given by:

.. math::

	P(D_i | T_i) \propto \prod_j \frac{\textrm{exp}\left(-\frac{\Delta\textbf{r}_j^2}{4\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}\right)}{4\pi\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}

The *D* inference mode is well-suited to freely diffusing molecules and the rapid characterization of the diffusivity.

This mode supports the :ref:`Jeffreys' prior <inference_jeffreys>` and the :ref:`diffusivity smoothing prior <inference_smoothing>` using the *smooth.d* mode instead of *d*.

.. _inference_df:

*DF* inference
^^^^^^^^^^^^^^

This inference mode estimates the diffusivity and force.
It takes advantage of the assumption :math:`D(\textbf{r}) \propto \frac{1}{\gamma(\textbf{r})}`.

The posterior probability used to infer the local diffusivity :math:`D_i` and force :math:`\textbf{F}_i` is given by:

.. math::

	P(D_i, \textbf{F}_i | T_i) \propto \prod_j \frac{\textrm{exp}\left(-\frac{\left(\Delta\textbf{r}_j - \frac{D_i\textbf{F}_i\Delta t_j}{k_BT}\right)^2}{4\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}\right)}{4\pi\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}

The *DF* inference mode is well-suited to mapping local force components, especially in the presence of non-potential forces (e.g. a rotational component).
This mode allows for the rapid characterization of the diffusivity and directional biases of the trajectories.

This mode supports the :ref:`Jeffreys' prior <inference_jeffreys>` and the :ref:`diffusivity smoothing prior <inference_smoothing>` using the *smooth.df* mode instead of *df*.

.. _inference_dd:

*DD* inference
^^^^^^^^^^^^^^

*DD* stands for *Diffusivity and Drift*.

This mode is very similar to the :ref:`DF mode <inference_df>` mode. 
The whole drift :math:`\frac{\textbf{F}}{\gamma}` is optimized instead of the force :math:`\textbf{F}`. 
This may offer increased stability in the optimization. 
Indeed the contribution of the drift to the objective function does not depend directly on the simultaneously explored diffusivity.

The maximized posterior probability is given by:

.. math::

	P(D_i, \frac{\textbf{F}_i}{\gamma_i} | T_i) \propto \prod_j \frac{\textrm{exp}\left(-\frac{\left(\Delta\textbf{r}_j - \frac{\textbf{F}_i}{\gamma_i}\Delta t_j/k_BT\right)^2}{4\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}\right)}{4\pi\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}

Although the force :math:`\textbf{F}_i` and friction coefficient :math:`\gamma_i` appear in the above expression, they are not explicitly evaluated. 
The drift :math:`\frac{\textbf{F}_i}{\gamma_i}` is treated as an indivisible variable.

The *DD* inference mode is well-suited to active processes (e.g. active transport phenomena).

This mode supports the :ref:`Jeffreys' prior <inference_jeffreys>` and the :ref:`diffusivity smoothing prior <inference_smoothing>` using the *smooth.dd* mode instead of *dd*.


.. _inference_dv:

*DV* inference
^^^^^^^^^^^^^^

The posterior probability used to infer the local diffusivity :math:`D_i` and potential energy :math:`V_i` is given by:

.. math::

	P(D_i, V_i | T_i) \propto \prod_j \frac{\textrm{exp}\left(-\frac{\left(\Delta\textbf{r}_j + \frac{D_i\nabla V_i\Delta t_j}{k_BT}\right)^2}{4\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}\right)}{4\pi\left(D_i+\frac{\sigma^2}{\Delta t_j}\right)\Delta t_j}P_S(\textbf{D})P_S(\textbf{V})

:math:`P_S(\textbf{D})` and :math:`P_S(\textbf{V})` are smoothing factors for the diffusivity and potential energy respectively.
The :math:`P_S(\textbf{D})` smoothing factor is also available for the other inference modes.
These factors are described in a :ref:`dedicated section <inference_smoothing>`.

This mode supports the :ref:`Jeffreys' prior <inference_jeffreys>`.


Priors and default values
^^^^^^^^^^^^^^^^^^^^^^^^^

All the methods use :math:`\sigma = 0.03 \textrm{µm}` as default value for the experimental localization error.
This parameter can be set with the ``-e`` command-line option or the `localization_error` argument to :func:`~tramway.helper.inference.infer` and is expressed in |um|.
Compare::

	> tramway -i example.rwa infer dd -e 0.01 -l DD_sigma_10nm

.. code-block:: python

	from tramway.helper import infer

	infer('example.rwa', 'DD', localization_error=0.01, output_label='DD_sigma_10nm')


Although not clearly indicated elsewhere, the diffusivity is bounded to the minimum value :math:`0` by default. 
If the Jeffreys' prior is requested, then this minimum default value is :math:`0.01`. 
This can be overwritten with the ``--min-diffusivity`` command-line option or the `min_diffusivity` argument to :func:`~tramway.helper.inference.infer`.

If no specific prior is defined, a uniform prior is used by default.

.. _inference_jeffreys:

Jeffreys' prior
"""""""""""""""

All the methods described here also feature an optional Jeffreys' prior on the diffusivity. 
It is a non-informative prior used to ensure that the posterior probability distribution is invariant by re-parametrization.

This prior - referred to as :math:`P_J(D_i)` - multiplies with the original expression of the posterior probability.
The maximized probability becomes:

.. math::

	P^*(D_i, ... | T_i) = P(D_i, ... | T_i) P_J(D_i)

Its value varies depending on the inference mode. Compare:

.. list-table:: Jeffreys' prior for the different inference modes
   :header-rows: 1

   * - Inference mode
     - Jeffreys' prior :math:`P_J(D_i)`

   * - :ref:`D <inference_d>`
     - :math:`\frac{1}{\left(D_i\overline{\Delta t}_i + \sigma^2\right)^2}`

   * - :ref:`DF <inference_df>`
     - :math:`\frac{D_i^2}{\left(D_i\overline{\Delta t}_i + \sigma^2\right)^2}`

   * - :ref:`DD <inference_dd>`
     - :math:`\frac{1}{\left(D_i\overline{\Delta t}_i + \sigma^2\right)^2}`

   * - :ref:`DV <inference_dv>`
     - :math:`\frac{D_i^2}{\left(D_i\overline{\Delta t}_i + \sigma^2\right)^2}`


The Jeffreys' prior may be introduced in the posterior probability with the ``-j`` command-line option or the `jeffreys_prior` argument to :func:`~tramway.helper.inference.infer`.
Compare::

	> tramway -i example.rwa infer dd -j -l DD_jeffreys

.. code-block:: python

	from tramway.helper import infer

	infer('example.rwa', 'DD', jeffreys_prior=True, output_label='DD_jeffreys')


Note that with this prior the default minimum diffusivity value is :math:`0.01`. 
Consider modifying this value.


.. _inference_smoothing:

Smoothing priors
""""""""""""""""

A smoothing (improper) prior penalizes the gradients of the inferred parameters. 
It is meant to reinforce the physical plausibility of the inferred maps. 
For example, in certain situations we do not expect large changes in the diffusion coefficient between neighbouring cells.

|tramway| features variants for the :ref:`D <inference_d>`, :ref:`DF <inference_df>` and :ref:`DD <inference_dd>` inference modes that add a smoothing factor :math:`P_S(\textbf{D})` for the diffusivity.
These variants are available as the respective alternative plugins: *smooth.d*, *smooth.df*, *smooth.dd*.

This prior multiplies with the original expression of the posterior probability and penalizes all the diffusivity gradients. 
:math:`P_S` is a function of the diffusivity at all the cells, hence the vectorial notation :math:`\textbf{D}` for the diffusivity.

The maximized probability becomes:

.. math::

	P^*(D_i, ... | T_i) = P(D_i, ... | T_i) P_S(\textbf{D})

with:

.. math::

	P_S(\textbf{D}) = \textrm{exp}\left(-\mu\sum_i ||\nabla D_i||^2\right)


The :math:`\mu` parameter can be set with the ``-d`` command-line option or the `diffusivity_prior` argument to :func:`~tramway.helper.inference.infer`.
Compare::

	> tramway -i example.rwa infer smooth.dd -d 1 -l DD_d_1

.. code-block:: python

	from tramway.helper import infer

	infer('example.rwa', 'smooth.dd', diffusivity_prior=1., output_label='DD_d_1')



Note that the :ref:`DV <inference_dv>` inference mode readily features this smoothing factor, in addition to a similar smoothing factor :math:`P_S(\textbf{V})` for the potential energy:

.. math::

	P_S(\textbf{V}) = \textrm{exp}\left(-\lambda\sum_i ||\nabla V_i||^2\right)

Similarly to :math:`\mu`, the :math:`\lambda` parameter can be set with the ``-v`` command-line option or the `potential_prior` argument to :func:`~tramway.helper.inference.infer`.


.. Advanced usage
.. --------------

.. Fuzzy cell-point association
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. Custom gradient
.. ^^^^^^^^^^^^^^^


