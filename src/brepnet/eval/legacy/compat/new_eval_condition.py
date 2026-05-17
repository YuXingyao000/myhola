"""Legacy compatibility wrapper.

The maintained implementation is `src.brepnet.eval.metrics.condition`.
"""

from src.brepnet.eval.legacy.compat.eval_condition import *  # noqa: F401,F403


if __name__ == "__main__":
    from src.brepnet.eval.legacy.compat.eval_condition import main

    main()
