
def _require_attrs(obj: Any, attrs: list[str], *, where: str) -> None:
    missing = [a for a in attrs if getattr(obj, a, None) is None]
    if missing:
        raise AttributeError(f"{where} is missing required attributes: {missing}")
