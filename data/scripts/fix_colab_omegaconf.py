#!/usr/bin/env python3
"""
Monkeypatch fairseq.dataclass.initialize to handle _MISSING_TYPE fields
before OmegaConf tries to validate them. Run this BEFORE importing fairseq.
"""

import sys
import dataclasses as _dc
from hydra.core.config_store import ConfigStore

# Patch hydra_init BEFORE fairseq imports it
def patched_hydra_init(cfg_name="config"):
    """Patched version that handles _MISSING_TYPE gracefully."""
    from fairseq.dataclass.configs import FairseqConfig
    
    cs = ConfigStore.instance()
    cs.store(name=f"{cfg_name}", node=FairseqConfig)
    
    # Handle fields with _MISSING_TYPE defaults
    for k, f in FairseqConfig.__dataclass_fields__.items():
        v = f.default
        
        # If default is _MISSING_TYPE, try default_factory
        if isinstance(v, type(_dc.MISSING)):
            df = getattr(f, "default_factory", _dc.MISSING)
            if not isinstance(df, type(_dc.MISSING)):
                try:
                    v = df()
                    print(f"✓ {k}: called default_factory, got {type(v).__name__}")
                except Exception as e:
                    print(f"✗ {k}: default_factory failed, using None. Error: {e}")
                    v = None
            else:
                print(f"✓ {k}: no default_factory, using None")
                v = None
        
        # Now store with concrete value
        try:
            cs.store(name=k, node=v)
        except Exception as e:
            print(f"✗ ERROR storing {k}: {e}")
            raise

# Apply the patch BEFORE any fairseq imports
print("[PATCH] Applying OmegaConf workaround...")

if "fairseq.dataclass.initialize" not in sys.modules:
    print("[PATCH] Fairseq not yet imported, patching early...")
    # Import and patch
    from fairseq.dataclass import initialize
    initialize.hydra_init = patched_hydra_init
    print("[PATCH] ✓ hydra_init replaced")
else:
    print("[PATCH] Fairseq already imported, attempting late patch...")
    sys.modules["fairseq.dataclass.initialize"].hydra_init = patched_hydra_init
    print("[PATCH] ✓ hydra_init replaced in sys.modules")

print("[PATCH] Ready to import fairseq")
