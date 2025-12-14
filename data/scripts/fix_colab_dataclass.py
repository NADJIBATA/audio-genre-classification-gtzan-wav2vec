#!/usr/bin/env python3
"""
Fix Python 3.12 dataclass mutable defaults error in Fairseq configs.py
Run this in Colab after uploading the repo.
"""

import os
import re
import shutil

# Path to configs.py in Colab environment
CONFIGS_PATH = "/content/fairseq/fairseq/dataclass/configs.py"

def fix_dataclass_defaults():
    """Replace mutable defaults X() with field(default_factory=X) in FairseqConfig."""
    
    if not os.path.exists(CONFIGS_PATH):
        print(f"ERROR: {CONFIGS_PATH} not found!")
        print("Make sure you've uploaded the repo and fairseq is installed.")
        return False
    
    # Read the file
    with open(CONFIGS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Backup
    backup_path = CONFIGS_PATH + ".bak"
    shutil.copy(CONFIGS_PATH, backup_path)
    print(f"✓ Backup created: {backup_path}")
    
    # Check if import field is present
    if "from dataclasses import dataclass, field" not in content:
        print("✓ Adding 'field' to dataclasses import...")
        content = content.replace(
            "from dataclasses import dataclass",
            "from dataclasses import dataclass, field"
        )
    else:
        print("✓ 'field' import already present")
    
    # Find the FairseqConfig dataclass block and replace mutable defaults
    # Pattern: whitespace + field_name: ClassName = ClassName()
    # Only replace inside @dataclass class FairseqConfig
    
    replacements = [
        ("common: CommonConfig = CommonConfig()", "common: CommonConfig = field(default_factory=CommonConfig)"),
        ("common_eval: CommonEvalConfig = CommonEvalConfig()", "common_eval: CommonEvalConfig = field(default_factory=CommonEvalConfig)"),
        ("distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()", "distributed_training: DistributedTrainingConfig = field(default_factory=DistributedTrainingConfig)"),
        ("dataset: DatasetConfig = DatasetConfig()", "dataset: DatasetConfig = field(default_factory=DatasetConfig)"),
        ("optimization: OptimizationConfig = OptimizationConfig()", "optimization: OptimizationConfig = field(default_factory=OptimizationConfig)"),
        ("checkpoint: CheckpointConfig = CheckpointConfig()", "checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)"),
        ("bmuf: FairseqBMUFConfig = FairseqBMUFConfig()", "bmuf: FairseqBMUFConfig = field(default_factory=FairseqBMUFConfig)"),
        ("generation: GenerationConfig = GenerationConfig()", "generation: GenerationConfig = field(default_factory=GenerationConfig)"),
        ("eval_lm: EvalLMConfig = EvalLMConfig()", "eval_lm: EvalLMConfig = field(default_factory=EvalLMConfig)"),
        ("interactive: InteractiveConfig = InteractiveConfig()", "interactive: InteractiveConfig = field(default_factory=InteractiveConfig)"),
        ("ema: EMAConfig = EMAConfig()", "ema: EMAConfig = field(default_factory=EMAConfig)"),
    ]
    
    count = 0
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            count += 1
            print(f"✓ Fixed: {old.split('=')[0].strip()}")
    
    # Write back
    with open(CONFIGS_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"\n✓ SUCCESS! Fixed {count} mutable defaults in {CONFIGS_PATH}")
    print("\nNow restart the Colab runtime (Runtime > Restart runtime) and try again:")
    print("  fairseq-train data/fairseq --task audio_classification ...")
    return True

if __name__ == "__main__":
    fix_dataclass_defaults()
