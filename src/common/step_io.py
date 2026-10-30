"""STEP reading at an explicit precision; OCC is imported only when called."""


def read_step_at_precision(path, mode: int = 0, precision: float = 0.1):
    """Do not let validity.py's import-time STEP settings alter native reads."""
    from OCC.Core.Interface import Interface_Static
    from OCC.Extend.DataExchange import read_step_file
    if not path.is_file():
        raise FileNotFoundError(path)
    old_mode = Interface_Static.IVal("read.precision.mode")
    old_precision = Interface_Static.RVal("read.precision.val")
    try:
        Interface_Static.SetIVal("read.precision.mode", mode)
        Interface_Static.SetRVal("read.precision.val", precision)
        shape = read_step_file(str(path), as_compound=False, verbosity=False)
        if shape is None or shape.IsNull():
            raise ValueError(f"No readable shape: {path}")
        return shape
    finally:
        Interface_Static.SetIVal("read.precision.mode", old_mode)
        Interface_Static.SetRVal("read.precision.val", old_precision)
