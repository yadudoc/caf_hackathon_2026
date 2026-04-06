def visualize_results(candidates: list[dict], lead: dict):
    import io, math, base64
    import cairosvg
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from IPython.display import display, HTML

    valid = [c for c in candidates if "mol" in c]
    for c in valid:
        c.setdefault("strain_energy", 0)

    if not valid:
        print("No valid candidates to visualize.")
        return

    lead_name  = lead["lead_name"]
    max_strain = max(c["strain_energy"] for c in valid) or 1

    def mol_to_b64(mol, w=360, h=220) -> str:
        drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
        drawer.drawOptions().padding = 0.15
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().encode()
        png = cairosvg.svg2png(bytestring=svg, output_width=w, output_height=h)
        return base64.b64encode(png).decode()

    def card_html(c: dict) -> str:
        is_lead    = c["name"] == lead_name
        b64        = mol_to_b64(Chem.RemoveHs(c["mol"]))
        border     = "#7ef0d4" if is_lead else "#0e3a50"
        name_color = "#7ef0d4" if is_lead else "#c8e6f0"
        badge      = '<div class="badge">★ Lead candidate</div>' if is_lead else ""
        lip        = "✓" if c.get("lipinski_ok") else "✗"
        return (
            '<div class="mol-card' + (' lead' if is_lead else '') + '" style="border-color:' + border + '">'
            + '<div class="mol-img-wrap">'
            + '<img src="data:image/png;base64,' + b64 + '" width="360" height="220" style="display:block;width:100%;height:auto">'
            + '</div>'
            + '<div class="mol-body">'
            + badge
            + '<div class="mol-name" style="color:' + name_color + '">' + c["name"] + '</div>'
            + '<div class="mol-stats">'
            + '<span class="stat-item">MW <b>' + f'{c["mw"]:.0f}' + '</b></span>'
            + '<span class="stat-item">logP <b>' + f'{c["logp"]:.2f}' + '</b></span>'
            + '<span class="stat-item">TPSA <b>' + f'{c["tpsa"]:.1f}' + '</b></span>'
            + '<span class="stat-item">Lipinski <b>' + lip + '</b></span>'
            + '</div></div></div>'
        )

    def bar_html(c: dict) -> str:
        import math
        raw = c["strain_energy"]
        c["strain_energy"] = 0 if (raw is None or not math.isfinite(raw)) else raw

        is_lead   = c["name"] == lead_name
        pct       = int(c["strain_energy"] / max_strain * 100)
        bar_color = "#7ef0d4" if is_lead else "#1a4a60"
        star      = " ★" if is_lead else ""
        cls       = "bar-row lead" if is_lead else "bar-row"
        return (
            '<div class="' + cls + '">'
            + '<div class="bar-name">' + c["name"] + star + '</div>'
            + '<div class="bar-track"><div class="bar-fill" style="width:' + str(pct) + '%;background:' + bar_color + '"></div></div>'
            + '<div class="bar-val">' + str(c["strain_energy"]) + 'kcal/mol</div>'
            + '</div>'
        )

    # Pull lead candidate dict for the summary panel
    lead_c = next((c for c in valid if c["name"] == lead_name), valid[0])

    CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
.dr-root { background:#060d14; color:#c8e6f0; font-family:'Space Mono',monospace; padding:2rem; border-radius:12px; margin:1rem 0; }
.dr-header { border-bottom:1px solid #0e3a50; padding-bottom:1rem; margin-bottom:1.75rem; }
.dr-title { font-family:'Syne',sans-serif; font-size:22px; font-weight:800; color:#7ef0d4; margin:0 0 4px; }
.dr-subtitle { font-size:10px; color:#3a7a90; text-transform:uppercase; letter-spacing:2px; }
.mol-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:14px; margin-bottom:2rem; }
.mol-card { background:#0a1822; border:1px solid #0e3a50; border-radius:10px; overflow:hidden; }
.mol-card.lead { box-shadow:0 0 0 1px #7ef0d422 inset; }
.mol-img-wrap { background:#f5f7f8; }
.mol-body { padding:10px 12px 14px; }
.badge { display:inline-block; font-size:9px; letter-spacing:1px; text-transform:uppercase; background:#7ef0d418; color:#7ef0d4; border:1px solid #7ef0d440; border-radius:4px; padding:2px 7px; margin-bottom:7px; }
.mol-name { font-family:'Syne',sans-serif; font-size:13px; font-weight:700; line-height:1.3; margin-bottom:8px; }
.mol-stats { display:flex; gap:10px; flex-wrap:wrap; }
.stat-item { font-size:10px; color:#3a7a90; }
.stat-item b { color:#7ab8cc; font-weight:700; }
.section-label { font-size:10px; color:#3a7a90; text-transform:uppercase; letter-spacing:2px; margin-bottom:12px; }
.bar-row { display:flex; align-items:center; gap:12px; margin-bottom:9px; }
.bar-name { font-size:10px; color:#3a7a90; width:200px; flex-shrink:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.bar-row.lead .bar-name { color:#7ef0d4; }
.bar-track { flex:1; height:18px; background:#0a1822; border:1px solid #0e3a50; border-radius:3px; overflow:hidden; }
.bar-fill { height:100%; border-radius:2px; }
.bar-val { font-size:10px; color:#7ab8cc; width:80px; text-align:right; flex-shrink:0; }
.lead-panel { background:#061812; border:1px solid #7ef0d430; border-radius:10px; padding:1.25rem; margin-top:1.75rem; display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; }
.lead-stat-label { font-size:10px; color:#3a7a90; text-transform:uppercase; letter-spacing:1px; margin-bottom:5px; }
.lead-stat-val { font-family:'Syne',sans-serif; font-size:22px; font-weight:800; color:#7ef0d4; }
.lead-stat-unit { font-size:10px; color:#3a7a90; margin-left:2px; }
.lead-reasoning { grid-column:1/-1; font-size:11px; color:#3a7a90; line-height:1.7; border-top:1px solid #0e3a50; padding-top:1rem; margin-top:0.5rem; }
.lead-reasoning b { color:#7ab8cc; font-weight:400; }
</style>
"""

    BODY = (
        '<div class="dr-root">'
        + '<div class="dr-header">'
        + '<div class="dr-title">EGFR Inhibitor Screen</div>'
        + '<div class="dr-subtitle">Drug Discovery · Binding Energy Analysis · ' + str(len(valid)) + ' Candidates</div>'
        + '</div>'
        + '<div class="mol-grid">' + "\n".join(card_html(c) for c in valid) + '</div>'
        + '<div class="section-label">Strain Energy Comparison · lower is better</div>'
        + '<div class="bar-chart">' + "\n".join(bar_html(c) for c in valid) + '</div>'
        + '<div class="lead-panel">'
        + '<div><div class="lead-stat-label">MW</div><div class="lead-stat-val">' + f'{lead_c["mw"]:.0f}' + '<span class="lead-stat-unit">Da</span></div></div>'
        + '<div><div class="lead-stat-label">logP</div><div class="lead-stat-val">' + f'{lead_c["logp"]:.2f}' + '</div></div>'
        + '<div><div class="lead-stat-label">TPSA</div><div class="lead-stat-val">' + f'{lead_c["tpsa"]:.1f}' + '<span class="lead-stat-unit">Ų</span></div></div>'
        + '<div><div class="lead-stat-label">Strain</div><div class="lead-stat-val">' + str(lead_c["strain_energy"]) + '<span class="lead-stat-unit">kal/mol</span></div></div>'
        + '<div class="lead-reasoning"><b>Lead: ' + lead_name + '</b><br>' + lead.get("reasoning", "") + '</div>'
        + '</div>'
        + '</div>'
    )

    display(HTML(CSS + BODY))
