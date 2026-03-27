import importlib
import inspect
import pkgutil
import unittest

from absl.testing import absltest

def get_all_modules(package_name):
    package = importlib.import_module(package_name)
    modules = [package]
    for _, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        try:
            modules.append(importlib.import_module(name))
        except ImportError:
            pass
    return set(modules)

class OverrideComplianceTest(unittest.TestCase):
    """Dynamically verifies that all `@typing.override` annotations are structurally valid across the repository."""

    def test_override_compliance(self):
        modules = get_all_modules('sequence_layers')
        checked_count = 0
        
        for mod in modules:
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                # Skip classes imported from other modules
                if getattr(obj, '__module__', None) != mod.__name__:
                    continue
                    
                for attr_name, attr_val in inspect.getmembers(obj):
                    if callable(attr_val) and getattr(attr_val, "__override__", False):
                        
                        # Find base method dynamically from MRO
                        base_class = None
                        base_method = None
                        for base in inspect.getmro(obj)[1:]: # skip self
                            if hasattr(base, attr_name):
                                base_class = base
                                base_method = getattr(base, attr_name)
                                break
                        
                        # Verify we are actually overriding something
                        if not base_method:
                            self.fail(f"Method '{obj.__name__}.{attr_name}' is annotated with @override but overrides nothing in base classes.")
                        
                        # Verify signature compatibility
                        try:
                            sig = inspect.signature(attr_val)
                            base_sig = inspect.signature(base_method)
                        except Exception:
                            continue # Skip non-inspectable builtins
                            
                        base_params = list(base_sig.parameters.values())
                        params = list(sig.parameters.values())
                        
                        # Ignore 'self' parameter
                        base_params = base_params[1:] if base_params else []
                        params = params[1:] if params else []
                        
                        by_name = {p.name: p for p in params}
                        
                        for bp in base_params:
                            if bp.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                                continue
                            
                            # Required explicitly named arguments cannot be dropped or renamed
                            if bp.name not in by_name:
                                if bp.kind != inspect.Parameter.POSITIONAL_ONLY:
                                    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
                                    if not has_kwargs:
                                        self.fail(f"Override mismatch in '{obj.__name__}.{attr_name}': missing or renamed parameter '{bp.name}'. Required by base class '{base_class.__name__}'.")
                                        
                        checked_count += 1
                        
        print(f"Verified {checked_count} explicit @override constraints dynamically at runtime.")

if __name__ == '__main__':
    absltest.main()
