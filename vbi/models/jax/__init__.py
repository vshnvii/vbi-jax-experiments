"""
JAX-based neural mass models and utilities.

Requirements:
    pip install vbi[jax]
"""

try:
    from . import base
    from . import integrators
    from . import noise
    from . import coupling
    from . import mpr
    from .mpr import JaxMPRModel, MPRParams
    
    # Also import and re-export models from vbjax
    try:
        import vbjax
        from vbjax import neural_mass as vbjax_neural_mass
    except ImportError:
        vbjax_neural_mass = None
    
    __all__ = ['base', 'integrators', 'noise', 'coupling', 'mpr', 'JaxMPRModel', 'MPRParams', 'vbjax_neural_mass']
    
except ImportError as e:
    raise ImportError(
        "JAX models require JAX to be installed. "
        "Install with: pip install vbi[jax]"
    ) from e