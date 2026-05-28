"""Quiet import of kaggle_environments.

kaggle_environments dumps several long lists to stdout on first import (an
OpenSpiel game enumeration). The output comes from a C extension that writes
directly to fd 1, so plain `sys.stdout` redirection is insufficient — we
must redirect fd 1 and fd 2 at the OS level. Has no effect on agent stdout
during gameplay (the redirection is restored before this module finishes).
"""

import contextlib
import logging
import os
import sys


@contextlib.contextmanager
def _suppress_fd_output():
    """Redirect fd 1 (stdout) and fd 2 (stderr) to /dev/null at the OS level.

    Catches output from C extensions that bypass sys.stdout / sys.stderr.
    """
    # Flush Python-side buffers first.
    sys.stdout.flush()
    sys.stderr.flush()

    devnull_fd = os.open(os.devnull, os.O_RDWR)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull_fd)


with _suppress_fd_output():
    import kaggle_environments  # noqa: F401
    from kaggle_environments import make  # noqa: F401
    from kaggle_environments.envs.orbit_wars import orbit_wars as _ow_module  # noqa: F401

# Silence any remaining loggers that fire later.
for name in (
    "kaggle_environments",
    "kaggle_environments.envs.open_spiel_env.open_spiel_env",
    "LiteLLM",
):
    logging.getLogger(name).setLevel(logging.ERROR)

Planet = _ow_module.Planet
Fleet = _ow_module.Fleet
BOARD_SIZE = _ow_module.BOARD_SIZE
CENTER = _ow_module.CENTER
SUN_RADIUS = _ow_module.SUN_RADIUS
ROTATION_RADIUS_LIMIT = _ow_module.ROTATION_RADIUS_LIMIT
COMET_RADIUS = _ow_module.COMET_RADIUS
COMET_SPAWN_STEPS = _ow_module.COMET_SPAWN_STEPS
