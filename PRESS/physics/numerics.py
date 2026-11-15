"""Scoped numerical precision without changing the process-wide JAX settings."""
import jax


def float64_context(enabled=True):
    """Return the supported JAX context manager for float64 integration."""
    context = getattr(jax, 'enable_x64', None)
    if context is None:
        from jax.experimental import enable_x64 as context
    return context(enabled)
