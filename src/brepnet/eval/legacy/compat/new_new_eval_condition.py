"""Legacy compatibility wrapper.

The old 24-rotation draft has been moved to
`src.brepnet.eval.legacy.new_new_eval_condition_legacy`.  New code should use
`src.brepnet.eval.metrics.condition`, which evaluates identity rotation only.
"""

from src.brepnet.eval.legacy.compat.eval_condition import *  # noqa: F401,F403


if __name__ == "__main__":
    from src.brepnet.eval.legacy.compat.eval_condition import main

    main()
