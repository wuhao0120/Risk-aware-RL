#!/usr/bin/env python3
"""Dependency-free runner for the focused test functions in this directory."""
import importlib
import inspect
import tempfile
from pathlib import Path
import traceback

MODULES = [
    "test_risk_semantics",
    "test_dqc_online",
    "test_dqc_finite",
    "test_dqc_gae",
    "test_weibull_checkpoint",
    "test_evaluation_exact",
    "test_actor_lr_schedule",
    "test_critic_calibration",
    "test_joint_sweep",
    "test_formal_campaign",
    "test_binding_q23_campaign",
    "test_qcpo_ref_context_campaign",
    "test_dynamic_seed0_campaign",
]

def main():
    total = failed = 0
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        for name, function in sorted(vars(module).items()):
            if not name.startswith("test_") or not callable(function):
                continue
            total += 1
            try:
                parameters = inspect.signature(function).parameters
                if "tmp_path" in parameters:
                    with tempfile.TemporaryDirectory() as directory:
                        function(tmp_path=Path(directory))
                else:
                    function()
                print(f"PASS {module_name}.{name}")
            except Exception:
                failed += 1
                print(f"FAIL {module_name}.{name}")
                traceback.print_exc()
    print(f"RESULT total={total} passed={total-failed} failed={failed}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
