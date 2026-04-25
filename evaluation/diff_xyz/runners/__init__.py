"""Driver scripts invoked by Crucible project specs.

Each runner is a thin wrapper around the harness or SFT loop, configured
entirely via environment variables so the same code path works for:

  - Local sanity check (set env, run python evaluation/diff_xyz/runners/X.py)
  - RunPod fleet (project spec sets env via env_set + variant overrides)
  - Colab (notebook sets env, calls subprocess)

Runners write a `result.json` to the path in DIFFXYZ_OUT (default
``/workspace/project/result.json``) and print one summary line on stdout
of the form ``RESULT EM=<float> IoU=<float>`` so Crucible's stdout metric
parser picks it up.
"""
