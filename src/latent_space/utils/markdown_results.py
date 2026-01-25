from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def flatten_dict(
    obj: Mapping[str, Any],
    *,
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """
    Flatten nested dictionaries:
      {"a": {"b": 1}, "c": 2} -> {"a.b": 1, "c": 2}
    """
    out: dict[str, Any] = {}
    for k, v in obj.items():
        key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, Mapping):
            out.update(flatten_dict(v, parent_key=key, sep=sep))
        else:
            out[key] = v
    return out


def to_plain_dict(obj: Any) -> dict[str, Any]:
    """
    Convert common config objects to dict:
    - dataclasses
    - objects with .dict() (pydantic-ish)
    - already dict
    """
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "dict") and callable(obj.dict):
        return obj.dict()  # type: ignore[no-any-return]
    # best-effort: use __dict__ if present
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    raise TypeError(f"Unsupported object type for to_plain_dict: {type(obj)!r}")


def _format_float(x: float, *, precision: int = 6) -> str:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    # compact but stable
    return f"{x:.{precision}g}"


def md_escape(text: str) -> str:
    """
    Escape pipes/newlines so markdown tables don't break.
    """
    return text.replace("|", "\\|").replace("\n", "\\n").replace("\r", "")


def to_cell(value: Any, *, float_precision: int = 6, max_len: int | None = 300) -> str:
    """
    Convert a Python value to a single markdown-table-safe string cell.
    """
    if value is None:
        s = ""
    elif isinstance(value, bool):
        s = "true" if value else "false"
    elif isinstance(value, (int,)):
        s = str(value)
    elif isinstance(value, float):
        s = _format_float(value, precision=float_precision)
    elif isinstance(value, (Path,)):
        s = str(value)
    elif isinstance(value, datetime):
        s = value.isoformat(timespec="seconds")
    elif isinstance(value, (list, tuple, set)):
        # JSON-ish for readability
        s = json.dumps(list(value), ensure_ascii=False)
    elif isinstance(value, Mapping):
        s = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        s = str(value)

    if max_len is not None and len(s) > max_len:
        s = s[: max_len - 1] + "…"

    return md_escape(s)


def _normalize_headers(headers: Sequence[str]) -> list[str]:
    # keep stable order; strip whitespace
    return [h.strip() for h in headers if h.strip()]


def parse_markdown_table(md: str) -> tuple[list[str], list[dict[str, str]]]:
    """
    Parse a simple markdown table of the form:
      | a | b |
      |---|---|
      | 1 | 2 |

    Returns (headers, rows), where each row maps header->cell (string).
    If no table found, returns ([], []).
    """
    lines = [ln.strip() for ln in md.splitlines() if ln.strip()]
    if len(lines) < 2:
        return ([], [])

    # find the first header line that looks like a table
    # very simple: first line contains '|' and second line contains '---'
    for i in range(len(lines) - 1):
        if (
            "|" in lines[i]
            and "|" in lines[i + 1]
            and set(lines[i + 1].replace("|", "").strip()) <= set("-: ")
        ):
            header_line = lines[i]
            sep_line = lines[i + 1]
            # ensure separator has at least one dash
            if "-" not in sep_line:
                continue

            headers = [h.strip() for h in header_line.strip("|").split("|")]
            headers = _normalize_headers(headers)

            rows: list[dict[str, str]] = []
            for ln in lines[i + 2 :]:
                if "|" not in ln:
                    break
                cells = [c.strip() for c in ln.strip("|").split("|")]
                # pad/truncate to header length
                if len(cells) < len(headers):
                    cells += [""] * (len(headers) - len(cells))
                if len(cells) > len(headers):
                    cells = cells[: len(headers)]
                rows.append({h: c for h, c in zip(headers, cells, strict=False)})
            return (headers, rows)

    return ([], [])


def render_markdown_table(
    headers: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    float_precision: int = 6,
) -> str:
    """
    Render headers + rows into a markdown table string.
    """
    headers = _normalize_headers(list(headers))
    if not headers:
        return ""

    # header row
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"

    body_lines: list[str] = []
    for r in rows:
        line_cells = [to_cell(r.get(h, ""), float_precision=float_precision) for h in headers]
        body_lines.append("| " + " | ".join(line_cells) + " |")

    return "\n".join([head, sep] + body_lines) + "\n"


class MarkdownTableLogger:
    """
    Append rows to a markdown table file, preserving existing rows/headers.

    Behavior:
    - If file doesn't exist: creates it with provided headers
    - If file exists: reads headers; if new keys appear, can auto-expand columns
    """

    def __init__(
        self,
        path: Path,
        *,
        headers: Sequence[str] | None = None,
        auto_expand_columns: bool = True,
        float_precision: int = 6,
    ) -> None:
        self.path = Path(path)
        self.float_precision = float_precision
        self.auto_expand_columns = auto_expand_columns
        self._headers = list(headers) if headers is not None else None

    @property
    def headers(self) -> list[str]:
        if self._headers is None:
            return []
        return list(self._headers)

    def _load(self) -> tuple[list[str], list[dict[str, str]]]:
        if not self.path.exists():
            return (self.headers, [])
        md = self.path.read_text(encoding="utf-8")
        headers, rows = parse_markdown_table(md)
        return (headers, rows)

    def prepend(self, row: Mapping[str, Any]) -> None:
        headers, rows = self._load()

        if not headers:
            if self._headers is None:
                headers = list(row.keys())
            else:
                headers = list(self._headers)

        if self.auto_expand_columns:
            for k in row.keys():
                if k not in headers:
                    headers.append(k)

        rows_any: list[dict[str, Any]] = [dict(row)]
        rows_any.extend(dict(r) for r in rows)

        out = render_markdown_table(headers, rows_any, float_precision=self.float_precision)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(out, encoding="utf-8")

    def append(self, row: Mapping[str, Any]) -> None:
        headers, rows = self._load()

        # if no table exists yet
        if not headers:
            if self._headers is None:
                # infer from row keys
                headers = list(row.keys())
            else:
                headers = list(self._headers)

        if self.auto_expand_columns:
            for k in row.keys():
                if k not in headers:
                    headers.append(k)

        # store as raw mapping; render handles formatting
        rows_any: list[dict[str, Any]] = [dict(r) for r in rows]  # existing are strings
        rows_any.append(dict(row))

        out = render_markdown_table(headers, rows_any, float_precision=self.float_precision)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(out, encoding="utf-8")


def merge_row(
    *parts: Mapping[str, Any],
    flatten: bool = True,
    sep: str = ".",
) -> dict[str, Any]:
    """
    Merge multiple dict-like parts into one row. Later parts override earlier parts.
    Optionally flatten nested dicts for each part.
    """
    merged: dict[str, Any] = {}
    for p in parts:
        d = dict(p)
        if flatten:
            d = flatten_dict(d, sep=sep)
        merged.update(d)
    return merged


def config_to_row(config: Any, *, flatten: bool = True, sep: str = ".") -> dict[str, Any]:
    d = to_plain_dict(config)
    return flatten_dict(d, sep=sep) if flatten else d
