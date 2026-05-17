import importlib.util


CONFIRM_CHAMFERDIST_NO_AVAILABLE = False
if importlib.util.find_spec("chamferdist") is None or CONFIRM_CHAMFERDIST_NO_AVAILABLE:
    CHAMFERDIST_AVAILABLE = False
    from src.brepnet.post.chamferdist_torch import ChamferDistanceTorch as ChamferDistance
else:
    CHAMFERDIST_AVAILABLE = True
    from chamferdist import ChamferDistance
