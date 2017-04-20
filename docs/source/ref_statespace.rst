StateSpace
==========

.. toctree::
    :maxdepth: 2
    :numbered:

State estimation of models with known parameters. Includes filtering/smoothing algorithms and maximum likelihood estimation of parameters with restrictions. State space model take the form 

.. math::

   y_t &= Z_t \alpha_t + d_t + \varepsilon_t  & \qquad & \varepsilon_t \sim N(0, H_t) \\
   \alpha_{t+1} &= T_{t+1} \alpha_t + c_{t+1} + R_{t+1} \eta_{t+1} & \qquad & \eta_t \sim N(0, Q_t) 

StateSpace Properties:
  Z, d, H - Observation equation parameters
  T, c, R, Q - Transition equation parameters
  a0, P0 - Initial value parameters

Object construction
-------------------

.. code-block:: matlab

   ss = StateSpace(Z, d, H, T, c, R, Q)

The d and c parameters may be entered as empty for convenience.

Time varrying parameters may be passed as structures with a field for the parameter values and a vector indicating the timing. Using Z as an example the field names would be Z.Zt and Z.tauZ.

Filtering and smoothing
-----------------------

.. sourcecode:: matlab

   [a, logl, filterOut] = ss.filter(y)

   [alpha, smootherOut] = ss.smooth(y)

Additional estimates from the filter (P, v, F, K) and smoother (eta, r, N, a0tilde) are returned in the filterOut and smootherOut structures. 

When the initial value parameters are not passed or are empty, default values will be generated as either the stationary solution (if it exists) or the approximate diffuse case with large kappa. The value of kappa can be set with ss.kappa before the use of the filter/smoother.

Mex versions will be used unless ss.useMex is set to false or the mex files cannot be found.

  
Method Reference
----------------
.. automodule:: src

.. _StateSpace:

.. autoclass:: StateSpace
    :show-inheritance:
    :members:
