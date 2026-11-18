"""Model-based near-field beam design, Bayesian tracking and link simulation.

The package replaces the learned one-pass Airy predictor of ``direct_network``
with an exact linear channel operator.  Every beam family (focus, Airy,
element-wise phase) is scored and optimized with the same operator, so the
online design generalizes to any geometry the operator is built for.
"""
