"""Scoped precision across JAX versions used locally and on Orfeo."""
import jax

def enable_x64(enabled=True):
    """Return the public precision context, with support for older JAX.

    JAX 0.9 removed the experimental alias. Resolve lazily so modern runtimes
    never access it; older runtimes keep their identical scoped behavior.
    """
    context = getattr(jax, 'enable_x64', None)
    if context is None:
        from jax.experimental import enable_x64 as context
    return context(enabled)
