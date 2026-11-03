from pathlib import Path


def append_result_row(path: Path, heading: str, row: str) -> None:
    """Insert `row` at the end of the first Markdown table after the `heading` line."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        raise ValueError(f"{path.name}: no heading {heading!r}") from None
    table_end = None
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("|"):
            table_end = i + 1
        elif table_end is not None or lines[i].startswith("#"):
            break
    if table_end is None:
        raise ValueError(f"{path.name}: no table under {heading!r}")
    if not lines[table_end - 1].endswith("\n"):
        lines[table_end - 1] += "\n"
    lines.insert(table_end, row.rstrip("\n") + "\n")
    path.write_text("".join(lines), encoding="utf-8")
