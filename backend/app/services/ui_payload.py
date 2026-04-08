import re
from typing import Optional


_NUM_RE = re.compile(r"^-?\d[\d,]*(?:\.\d+)?$")


def _to_number(value: str) -> Optional[float]:
    cleaned = (value or "").strip().replace(",", "")
    if not _NUM_RE.match(cleaned):
        return None
    try:
        return float(cleaned)
    except Exception:
        return None


def _parse_markdown_table(answer: str):
    lines = [line.strip() for line in (answer or "").splitlines() if "|" in line]
    if len(lines) < 3:
        return None

    header_cells = [c.strip() for c in lines[0].strip("|").split("|")]
    sep_line = lines[1].strip().replace(" ", "")
    if "-" not in sep_line:
        return None

    rows = []
    for row_line in lines[2:]:
        cells = [c.strip() for c in row_line.strip("|").split("|")]
        if len(cells) != len(header_cells):
            continue
        rows.append(cells)
    if not rows or len(header_cells) < 2:
        return None

    return header_cells, rows


def build_ui_payload(question: str, answer: str) -> Optional[dict]:
    q = (question or "").lower()
    if not any(k in q for k in ("compare", "comparison", "trend", "q1", "q2", "q3", "q4", "financial", "chart", "graph")):
        return None

    parsed = _parse_markdown_table(answer)
    if not parsed:
        return None
    headers, rows = parsed

    category_key = headers[0]
    value_headers = headers[1:]
    datasets = [{"label": h, "data": []} for h in value_headers]
    labels = []

    for row in rows:
        labels.append(row[0])
        for idx, value in enumerate(row[1:]):
            numeric = _to_number(value)
            datasets[idx]["data"].append(numeric if numeric is not None else 0.0)

    if not labels:
        return None

    return {
        "type": "chart",
        "chart_type": "bar",
        "title": "Generated comparison",
        "x_axis": category_key,
        "labels": labels,
        "datasets": datasets,
    }
