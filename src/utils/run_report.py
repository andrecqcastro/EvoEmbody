"""
Console summaries during a run, and the final report: summary.txt,
summary.json, and the lineage graph (evolution_graph.html / .png).
"""
import html
import json
import os

from src.utils.run_context import record_fallback

# Statuses look like discarded_check5, capped_tier60 or reached_tier140 and are
# styled by prefix.
def status_style(status: str) -> tuple:
    if status.startswith("discarded"):
        return ("#9b2226", "#fbe6e6")
    if status.startswith("capped"):
        return ("#a86a00", "#fdf1dd")
    if status.startswith("reached"):
        return ("#1f7a3f", "#e6f4ec")
    return ("#3f5b8c", "#e8eefa")  # alive / unknown


def status_label(status: str) -> str:
    if status == "alive":
        return "not evaluated"
    if status.startswith("discarded_"):
        return f"discarded at {status[len('discarded_'):]}"
    if status.startswith("capped_"):
        return f"capped at {status[len('capped_'):]}"
    if status.startswith("reached_"):
        return f"reached {status[len('reached_'):]}"
    return status


# ---- concise summaries printed during the run ------------------------------

def _first_paragraph(text: str, marker: str, limit: int = 700) -> str:
    """The text after `marker`, up to the next blank line or heading."""
    upper = text.upper()
    start = upper.find(marker)
    if start == -1:
        body = text.strip()
    else:
        body = text[start + len(marker):].lstrip(": \n")
        for stop in ("\nDIRECT ERRORS", "\nPOTENTIAL ERRORS", "\n\n"):
            cut = body.upper().find(stop.upper()) if stop.isupper() else body.find(stop)
            if cut > 0:
                body = body[:cut]
    body = " ".join(body.split())
    return body[:limit] + ("…" if len(body) > limit else "")


def summarize_problem(report_text: str) -> str:
    return _first_paragraph(report_text, "ROOT CAUSE")


def summarize_solution(plan: dict, plan_text: str) -> str:
    if not plan:
        return _first_paragraph(plan_text, "ROOT_CAUSE")
    changes = "; ".join(
        f"{os.path.basename(entry.get('file_path', '?'))}: {entry.get('description', '')}"
        for entry in plan.get("mutation_plan", []))
    cause = " ".join(str(plan.get("root_cause", "")).split())
    return f"{cause[:500]} -> {changes}" if cause else changes


def print_meta_summary(cycle: int, parent_id: str, problem: str, solution: str,
                       fallbacks: int, tokens: int) -> None:
    print(f"[meta agents] cycle {cycle} on parent {parent_id}")
    print(f"  problem : {problem or '(none reported)'}")
    print(f"  solution: {solution or '(none reported)'}")
    print(f"  {tokens:,} tokens"
          + (f" | {fallbacks} no-think fallback(s)" if fallbacks else ""))


# ---- final report ----------------------------------------------------------

def _node_rows(benchmark, archive: dict) -> list:
    rows = []
    for node in archive["nodes"]:
        stage = benchmark.highest_completed_stage(node)
        info = node["stages"].get(stage, {}) if stage else {}
        bonus_sources = node.get("bonus_sources", {})
        bonus_values = node.get("bonus_values", {})
        rows.append({
            "id": node["id"],
            "parent": node["parent_id"] or "-",
            "cycle": "bootstrap" if node["cycle_index"] < 0 else str(node["cycle_index"] + 1),
            "status": node["status"],
            "stage": stage or "-",
            "passed": f"{info.get('passed', '-')}/{info.get('n_tasks', '-')}",
            "score": node["score"],
            "efficiency": node["avg_step_efficiency"],
            "meta": node.get("mutation", {}).get("outcome", "-"),
            "fallbacks": node.get("no_think_fallbacks", 0),
            "tier_index": bonus_sources.get("tier_index", -1),
            "children_count": bonus_sources.get("children_count", 0),
            "tier_bonus": bonus_values.get("tier_bonus"),
            "steps_bonus": bonus_values.get("steps_bonus"),
        })
    return rows


def _summary_text(benchmark, archive: dict, rows: list, fallbacks: list, cost_log: list,
                  total_tokens: int, stop_reason: str, cycles: int) -> str:
    lines = [
        "=" * 120,
        "ALFWORLD EVOLUTION SUMMARY",
        "=" * 120,
        f"models        : {archive['models']}",
        f"game pool     : {archive['game_pool']['total_pool_size']} games, "
        f"stage plan {archive['game_pool']['stage_plan']}",
        f"cycles run    : {cycles}",
        f"tokens used   : {total_tokens:,}",
        f"stopped       : {stop_reason}",
        "",
        f"{'node':<13}{'parent':<13}{'cycle':<10}{'stage':<9}{'passed':<9}"
        f"{'score':<8}{'step eff':<10}{'tier bonus':<11}{'meta agents':<28}{'status'}",
        "-" * 130,
    ]
    for row in rows:
        score = f"{row['score']:.3f}" if row["score"] is not None else "-"
        efficiency = f"{row['efficiency']:.3f}" if row["efficiency"] is not None else "-"
        tier_bonus = f"{row['tier_bonus']:.3f}" if row["tier_bonus"] is not None else "-"
        meta = row["meta"] + (f" (+{row['fallbacks']} no-think)" if row["fallbacks"] else "")
        lines.append(f"{row['id']:<13}{row['parent']:<13}{row['cycle']:<10}{row['stage']:<9}"
                     f"{row['passed']:<9}{score:<8}{efficiency:<10}{tier_bonus:<11}{meta[:27]:<28}"
                     f"{status_label(row['status'])}")
    lines.append("-" * 130)

    scored = [n for n in archive["nodes"] if n["score"] is not None]
    if scored:
        best = benchmark.pick_best_node(scored)
        root = archive["nodes"][0]
        lines += [
            "",
            f"BEST NODE: {best['id']}, score {best['score']:.3f}, "
            f"stage {benchmark.highest_completed_stage(best)}, "
            f"step efficiency {best['avg_step_efficiency']}",
            f"  genome: {best['agent_json_path']}",
        ]
        if best["id"] != root["id"] and root["score"] is not None:
            lines.append(f"  root {root['id']} scored {root['score']:.3f}, "
                         f"net {best['score'] - root['score']:+.3f}")

    lines += ["", "COST PER CYCLE (input/output tokens)",
             f"{'cycle':<38}{'meta in':>12}{'meta out':>12}"
             f"{'episodes in':>14}{'episodes out':>14}{'total':>14}"]
    for entry in cost_log:
        lines.append(f"{entry['label']:<38}{entry['meta_tokens_in']:>12,}"
                     f"{entry['meta_tokens_out']:>12,}{entry['episode_tokens_in']:>14,}"
                     f"{entry['episode_tokens_out']:>14,}{entry['total']:>14,}")

    lines += ["", f"AUTOMATIC FALLBACKS: {len(fallbacks)}"]
    counted = {}
    for event in fallbacks:
        counted[event["kind"]] = counted.get(event["kind"], 0) + 1
    lines += [f"  {kind}: {count}" for kind, count in sorted(counted.items())] or ["  none"]
    lines.append("=" * 120)
    return "\n".join(lines)


# ---- lineage graph ---------------------------------------------------------

BOX_W, BOX_H, GAP_X, GAP_Y, MARGIN = 190, 116, 40, 74, 40


def _layout(archive: dict) -> dict:
    """Assigns each node a depth from the root and a column within that depth."""
    depth = {}
    for node in archive["nodes"]:
        parent = node["parent_id"]
        depth[node["id"]] = 0 if parent is None else depth.get(parent, 0) + 1

    # Order each level by its parent's column, so siblings sit together and an
    # edge never has to cross the whole diagram.
    order = {}
    for node in archive["nodes"]:
        order.setdefault(depth[node["id"]], []).append(node["id"])

    positions, column_of = {}, {}
    for level in sorted(order):
        ids = sorted(order[level], key=lambda i: column_of.get(
            next(n["parent_id"] for n in archive["nodes"] if n["id"] == i), -1))
        for column, node_id in enumerate(ids):
            column_of[node_id] = column
            positions[node_id] = (MARGIN + column * (BOX_W + GAP_X),
                                  MARGIN + level * (BOX_H + GAP_Y))
    return positions


def _svg(benchmark, archive: dict) -> str:
    positions = _layout(archive)
    width = max((x for x, _ in positions.values()), default=0) + BOX_W + MARGIN
    height = max((y for _, y in positions.values()), default=0) + BOX_H + MARGIN

    edges, boxes = [], []
    for node in archive["nodes"]:
        x, y = positions[node["id"]]
        if node["parent_id"] in positions:
            px, py = positions[node["parent_id"]]
            edges.append(
                f'<path d="M{px + BOX_W / 2:.0f},{py + BOX_H} '
                f'C{px + BOX_W / 2:.0f},{py + BOX_H + GAP_Y / 2:.0f} '
                f'{x + BOX_W / 2:.0f},{y - GAP_Y / 2:.0f} '
                f'{x + BOX_W / 2:.0f},{y}" class="edge"/>')

        stroke, fill = status_style(node["status"])
        stage = benchmark.highest_completed_stage(node)
        info = node["stages"].get(stage, {}) if stage else {}
        score = f"{node['score']:.1%}" if node["score"] is not None else "not scored"
        efficiency = (f"{node['avg_step_efficiency']:.2f}"
                      if node["avg_step_efficiency"] is not None else "-")
        cycle = "bootstrap" if node["cycle_index"] < 0 else f"cycle {node['cycle_index'] + 1}"
        detail = (f"{info.get('passed', '-')}/{info.get('n_tasks', '-')} solved"
                  if info else status_label(node["status"]))
        bonus = node.get("bonus_values", {})
        bonus_line = (f"tier×{bonus.get('tier_bonus', 0):.2f} · steps×{bonus.get('steps_bonus', 0):.2f} "
                     f"· child×{bonus.get('children_factor', 0):.2f}")

        boxes.append(f'''<g class="node">
  <rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="10"
        fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <text x="{x + 12}" y="{y + 24}" class="id">{html.escape(node["id"])}</text>
  <text x="{x + BOX_W - 12}" y="{y + 24}" class="cycle" text-anchor="end">{cycle}</text>
  <text x="{x + 12}" y="{y + 48}" class="score" fill="{stroke}">{score}</text>
  <text x="{x + 12}" y="{y + 67}" class="meta">{html.escape(detail)} · {stage or "-"}</text>
  <text x="{x + 12}" y="{y + 84}" class="meta">step eff {efficiency} · {
      html.escape(status_label(node["status"]))}</text>
  <text x="{x + 12}" y="{y + 101}" class="meta">{html.escape(bonus_line)}</text>
</g>''')

    return (f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'xmlns="http://www.w3.org/2000/svg" role="img">\n'
            + "\n".join(edges + boxes) + "\n</svg>")


SCALE = 2                      # drawn at 2x, so the PNG stays sharp when zoomed
FONT_DIR = "/usr/share/fonts/truetype/dejavu"


def _fonts():
    from PIL import ImageFont

    def load(name, size):
        try:
            return ImageFont.truetype(os.path.join(FONT_DIR, name), size * SCALE)
        except OSError:
            return ImageFont.load_default()

    return {"id": load("DejaVuSans-Bold.ttf", 13), "cycle": load("DejaVuSans.ttf", 10),
            "score": load("DejaVuSans-Bold.ttf", 20), "meta": load("DejaVuSans.ttf", 10)}


def _png(benchmark, archive: dict, path: str) -> bool:
    """Draws the lineage graph as a PNG. Returns False when Pillow is not installed."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False

    positions = _layout(archive)
    width = (max((x for x, _ in positions.values()), default=0) + BOX_W + MARGIN) * SCALE
    height = (max((y for _, y in positions.values()), default=0) + BOX_H + MARGIN) * SCALE

    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    font = _fonts()
    s = lambda v: int(v * SCALE)

    for node in archive["nodes"]:
        if node["parent_id"] not in positions:
            continue
        x, y = positions[node["id"]]
        px, py = positions[node["parent_id"]]
        x0, y0 = px + BOX_W / 2, py + BOX_H
        x1, y1 = x + BOX_W / 2, y
        points = []
        for i in range(25):
            t = i / 24
            ease = t * t * (3 - 2 * t)
            points.append((s(x0 + (x1 - x0) * ease), s(y0 + (y1 - y0) * t)))
        draw.line(points, fill="#c9c9c9", width=s(2), joint="curve")

    for node in archive["nodes"]:
        x, y = positions[node["id"]]
        stroke, fill = status_style(node["status"])
        stage = benchmark.highest_completed_stage(node)
        info = node["stages"].get(stage, {}) if stage else {}
        score = f"{node['score']:.1%}" if node["score"] is not None else "not scored"
        efficiency = (f"{node['avg_step_efficiency']:.2f}"
                      if node["avg_step_efficiency"] is not None else "-")
        cycle = "bootstrap" if node["cycle_index"] < 0 else f"cycle {node['cycle_index'] + 1}"
        detail = (f"{info.get('passed', '-')}/{info.get('n_tasks', '-')} solved"
                  if info else status_label(node["status"]))
        bonus = node.get("bonus_values", {})
        bonus_line = (f"tier×{bonus.get('tier_bonus', 0):.2f} · steps×{bonus.get('steps_bonus', 0):.2f} "
                     f"· child×{bonus.get('children_factor', 0):.2f}")

        draw.rounded_rectangle([s(x), s(y), s(x + BOX_W), s(y + BOX_H)], radius=s(10),
                               fill=fill, outline=stroke, width=s(2))
        draw.text((s(x + 12), s(y + 12)), node["id"], font=font["id"], fill="#1a1a1a")
        draw.text((s(x + BOX_W - 12), s(y + 13)), cycle, font=font["cycle"], fill="#6a6a6a",
                  anchor="ra")
        draw.text((s(x + 12), s(y + 32)), score, font=font["score"], fill=stroke)
        draw.text((s(x + 12), s(y + 59)), f"{detail} · {stage or '-'}",
                  font=font["meta"], fill="#5a5a5a")
        draw.text((s(x + 12), s(y + 75)),
                  f"step eff {efficiency} · {status_label(node['status'])}",
                  font=font["meta"], fill="#5a5a5a")
        draw.text((s(x + 12), s(y + 92)), bonus_line, font=font["meta"], fill="#5a5a5a")

    image.save(path, "PNG")
    return True


LEGEND = (
    ("alive", "not evaluated"),
    ("discarded_check5", "discarded at the check5 gate"),
    ("capped_tier", "capped before the final tier"),
    ("reached_tier", "reached the final tier"),
)


def _graph_html(benchmark, archive: dict, summary: str) -> str:
    legend = "".join(
        f'<span class="key">'
        f'<i style="background:{status_style(status)[1]};border-color:{status_style(status)[0]}"></i>'
        f'{label}</span>'
        for status, label in LEGEND)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>ALFWorld evolution, {html.escape(archive.get('created_at', ''))}</title>
<style>
 :root {{ color-scheme: light dark; --fg:#1a1a1a; --bg:#ffffff; --muted:#5a5a5a; --line:#c9c9c9; }}
 @media (prefers-color-scheme: dark) {{
   :root {{ --fg:#ececec; --bg:#161616; --muted:#a8a8a8; --line:#4a4a4a; }}
 }}
 body {{ margin:0; padding:32px; background:var(--bg); color:var(--fg);
        font:15px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif; }}
 h1 {{ font-size:20px; margin:0 0 4px; }}
 p.sub {{ color:var(--muted); margin:0 0 24px; }}
 .legend {{ display:flex; gap:18px; flex-wrap:wrap; margin-bottom:18px; }}
 .key {{ display:flex; align-items:center; gap:7px; font-size:13px; color:var(--muted); }}
 .key i {{ width:14px; height:14px; border-radius:4px; border:2px solid; display:inline-block; }}
 .canvas {{ overflow-x:auto; border:1px solid var(--line); border-radius:12px; padding:8px; }}
 .edge {{ fill:none; stroke:var(--line); stroke-width:2; }}
 text {{ font-family:ui-sans-serif,system-ui,sans-serif; }}
 /* Node fills are always light, so text inside a box keeps fixed dark colours
    rather than following the page theme. */
 .id {{ font-size:14px; font-weight:650; fill:#1a1a1a; }}
 .cycle {{ font-size:11px; fill:#6a6a6a; }}
 .score {{ font-size:21px; font-weight:700; }}
 .meta {{ font-size:11px; fill:#5a5a5a; }}
 pre {{ margin-top:28px; padding:18px; border:1px solid var(--line); border-radius:12px;
        overflow-x:auto; font-size:12.5px; line-height:1.45; background:transparent; }}
</style></head><body>
<h1>ALFWorld evolution</h1>
<p class="sub">{html.escape(str(archive.get("models", {})))} · created {html.escape(archive.get("created_at", ""))}</p>
<div class="legend">{legend}</div>
<div class="canvas">{_svg(benchmark, archive)}</div>
<pre>{html.escape(summary)}</pre>
</body></html>"""


def write_final_report(benchmark, archive: dict, paths, fallbacks: list, cost_log: list,
                       total_tokens: int, stop_reason: str, cycles: int) -> None:
    rows = _node_rows(benchmark, archive)
    summary = _summary_text(benchmark, archive, rows, fallbacks, cost_log, total_tokens,
                            stop_reason, cycles)
    print("\n" + summary)

    with open(paths.summary_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    with open(paths.graph_path, "w", encoding="utf-8") as f:
        f.write(_graph_html(benchmark, archive, summary))
    if not _png(benchmark, archive, paths.graph_png_path):
        record_fallback("graph_png_skipped", reason="Pillow is not installed")
    total_tokens_in = sum(e["meta_tokens_in"] + e["episode_tokens_in"] for e in cost_log)
    total_tokens_out = sum(e["meta_tokens_out"] + e["episode_tokens_out"] for e in cost_log)
    with open(os.path.join(paths.run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"models": archive["models"], "stop_reason": stop_reason,
                   "cycles": cycles, "total_tokens": total_tokens,
                   "total_tokens_in": total_tokens_in, "total_tokens_out": total_tokens_out,
                   "nodes": rows, "cost_log": cost_log, "fallbacks": fallbacks}, f, indent=2)

    print(f"\nRun directory : {paths.run_dir}")
    print(f"Summary       : {paths.summary_path}")
    print(f"Lineage graph : {paths.graph_path}")
    if os.path.exists(paths.graph_png_path):
        print(f"                {paths.graph_png_path}")
