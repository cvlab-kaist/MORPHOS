import importlib

__attributes = {
    'SparseStructureLatent': 'sparse_structure_latent',
    'TextConditionedSparseStructureLatent': 'sparse_structure_latent',
    'ImageConditionedSparseStructureLatent': 'sparse_structure_latent',

    'SLat': 'structured_latent',
    'TextConditionedSLat': 'structured_latent',
    'ImageConditionedSLat': 'structured_latent',

    'TemporalSS':   'temporal_ss',
    'TemporalSLat': 'temporal_slat',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_latent import (
        SparseStructureLatent,
        TextConditionedSparseStructureLatent,
        ImageConditionedSparseStructureLatent,
    )

    from .structured_latent import (
        SLat,
        TextConditionedSLat,
        ImageConditionedSLat,
    )

    from .temporal_ss   import TemporalSS
    from .temporal_slat import TemporalSLat

