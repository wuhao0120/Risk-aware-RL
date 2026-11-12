"""Internal recurrent network, rollout, optimization and checkpoint components.

Only run.py (DCA), qcpo_run.py (QCPO), and run_tqpo.py (TQPO) are experiment
entrypoints. The final subclasses reuse these numerical kernels and validators;
legacy identifiers are retained to read the original scientific checkpoints.
Actual cost semantics, replay/PI settings and discounts are installed by the
final agents and recorded in their actual_* checkpoint identity fields.
"""
