drug_discovery_demo.py
python"""
LLM-guided drug discovery demo:
1. LLM generates candidate SMILES for a target
2. RDKit validates and prepares ligands
3. OpenMM estimates binding energy via minimization
4. Best candidate visualized with nglview
"""

import json
import anthropic
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import Draw
import matplotlib.pyplot as plt


# ── Step 1: LLM generates candidate SMILES ────────────────────

def generate_candidates(target_description: str, n: int = 5) -> list[dict]:
    """Ask the LLM to generate candidate drug-like SMILES for a target."""

    prompt = f"""You are a medicinal chemist AI.
Generate {n} small-molecule drug candidates targeting: {target_description}

Rules:
- Each molecule must be drug-like (Lipinski's Rule of Five)
- Prefer known pharmacophores for this target class
- Return ONLY valid JSON, no preamble, no markdown fences

Return a JSON array of objects with these fields:
  "smiles"      : valid SMILES string
  "name"        : short descriptive name
  "rationale"   : one sentence explaining why this scaffold is promising

Example format:
[
  {{"smiles": "CCO", "name": "ethanol", "rationale": "Simple example."}}
]
"""

    client   = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if model added them anyway
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1])

    candidates = json.loads(raw)
    print(f"✓ LLM generated {len(candidates)} candidates")
    for c in candidates:
        print(f"  {c['name']:30s}  {c['smiles']}")
    return candidates


# ── Step 2: Validate and featurize with RDKit ─────────────────

def validate_and_featurize(candidates: list[dict]) -> list[dict]:
    """Validate SMILES and compute drug-likeness descriptors."""
    valid = []
    for c in candidates:
        mol = Chem.MolFromSmiles(c["smiles"])
        if mol is None:
            print(f"  ✗ Invalid SMILES skipped: {c['name']}")
            continue

        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if result != 0:
            print(f"  ✗ 3D embedding failed: {c['name']}")
            continue
        AllChem.MMFFOptimizeMolecule(mol)

        c["mol"]        = mol
        c["mw"]         = Descriptors.MolWt(mol)
        c["logp"]       = Descriptors.MolLogP(mol)
        c["hbd"]        = Descriptors.NumHDonors(mol)
        c["hba"]        = Descriptors.NumHAcceptors(mol)
        c["tpsa"]       = Descriptors.TPSA(mol)
        c["n_atoms"]    = mol.GetNumAtoms()
        c["lipinski_ok"] = (
            c["mw"] <= 500 and c["logp"] <= 5 and
            c["hbd"] <= 5  and c["hba"] <= 10
        )
        valid.append(c)

    print(f"✓ {len(valid)}/{len(candidates)} candidates passed validation")
    return valid


# ── Step 3: OpenMM binding energy proxy ───────────────────────

def estimate_binding_energy(candidates: list[dict]) -> list[dict]:
    """
    Simplified binding energy proxy using OpenMM vacuum minimization.
    Real docking would use a protein PDB + docking engine (e.g. AutoDock-GPU).
    Here we compute strain energy (minimized - starting) as a proxy for
    ligand flexibility / binding readiness.
    """
    from openff.toolkit import Molecule
    from openff.forcefields import ForceField
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator
    from openmm import (
        System, LangevinMiddleIntegrator,
        Platform, Vec3, unit
    )
    from openmm.app import Simulation, ForceField as AppForceField
    from openmm.unit import (
        kilocalories_per_mole, kilojoules_per_mole,
        kelvin, femtoseconds, picosecond
    )

    for c in candidates:
        try:
            # Convert RDKit mol → OpenFF Molecule
            off_mol  = Molecule.from_rdkit(c["mol"], allow_undefined_stereo=True)
            ff       = ForceField("openff-2.0.0.offxml")
            topology = off_mol.to_topology()
            system   = ff.create_openmm_system(topology)

            positions = c["mol"].GetConformer().GetPositions()  # Angstrom
            positions_nm = positions * 0.1                       # → nm

            integrator = LangevinMiddleIntegrator(
                300 * kelvin, 1.0 / picosecond, 2.0 * femtoseconds
            )

            platform = None
            for name in ['CUDA', 'OpenCL', 'CPU']:
                try:
                    platform = Platform.getPlatformByName(name)
                    break
                except Exception:
                    continue

            sim = Simulation(
                topology.to_openmm(), system, integrator, platform
            )
            sim.context.setPositions(positions_nm * unit.nanometer)

            # Energy before minimization
            state_before = sim.context.getState(getEnergy=True)
            e_before = state_before.getPotentialEnergy().value_in_unit(
                kilojoules_per_mole
            )

            sim.minimizeEnergy(maxIterations=500)

            # Energy after minimization
            state_after = sim.context.getState(getEnergy=True)
            e_after = state_after.getPotentialEnergy().value_in_unit(
                kilojoules_per_mole
            )

            c["e_before"]     = round(e_before, 2)
            c["e_after"]      = round(e_after, 2)
            c["strain_energy"] = round(e_before - e_after, 2)

            print(f"  {c['name']:30s}  strain={c['strain_energy']:.1f} kJ/mol")

        except Exception as ex:
            print(f"  ✗ OpenMM failed for {c['name']}: {ex}")
            c["strain_energy"] = float("inf")

    return candidates


# ── Step 4: LLM ranks and selects lead ───────────────────────

def select_lead(candidates: list[dict]) -> dict:
    """Ask the LLM to reason over descriptors and pick the best lead."""

    summaries = [
        {
            "name":          c["name"],
            "smiles":        c["smiles"],
            "rationale":     c["rationale"],
            "mw":            c.get("mw"),
            "logp":          c.get("logp"),
            "tpsa":          c.get("tpsa"),
            "lipinski_ok":   c.get("lipinski_ok"),
            "strain_energy_kJ_mol": c.get("strain_energy"),
        }
        for c in candidates if "strain_energy" in c
    ]

    prompt = f"""You are an expert computational medicinal chemist.
Below are drug candidates with computed properties and strain energies.
Lower strain energy indicates a more geometrically relaxed, binding-ready ligand.

Pick the single best lead candidate, balancing:
- Low strain energy (flexible, ready to bind)
- Good drug-likeness (Lipinski compliant, low TPSA)
- Chemical novelty and rationale quality

Return ONLY valid JSON with fields:
  "lead_name"  : name of chosen candidate
  "reasoning"  : 2-3 sentence explanation of the choice

Candidates:
{json.dumps(summaries, indent=2)}
"""

    client   = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1])

    result = json.loads(raw)
    lead   = next(c for c in candidates if c["name"] == result["lead_name"])
    lead["llm_reasoning"] = result["reasoning"]

    print(f"\n✓ Lead selected: {lead['name']}")
    print(f"  {lead['llm_reasoning']}")
    return lead


# ── Step 5: Visualize lead + candidate grid ───────────────────

def visualize_results(candidates: list[dict], lead: dict):
    """
    Two-panel output:
    - Left:  2D structure grid of all candidates (RDKit)
    - Right: property radar + strain energy bar chart
    """
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    from IPython.display import display, Image as IPImage
    import io

    valid = [c for c in candidates if "mol" in c]

    # 2D grid
    mols   = [Chem.RemoveHs(c["mol"]) for c in valid]
    labels = [
        f"{c['name']}\nMW={c['mw']:.0f}  logP={c['logp']:.1f}\n"
        f"strain={c.get('strain_energy', '?')} kJ/mol"
        + (" ★ LEAD" if c["name"] == lead["name"] else "")
        for c in valid
    ]

    img = Draw.MolsToGridImage(
        mols, molsPerRow=3, subImgSize=(400, 300),
        legends=labels
    )

    # Bar chart of strain energies
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d1117')

    # Left: molecule grid (rendered via RDKit PNG → matplotlib)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    grid_img = plt.imread(buf)
    axes[0].imshow(grid_img)
    axes[0].axis('off')
    axes[0].set_title('Candidate Molecules', color='white', fontsize=13)

    # Right: strain energy bar chart
    ax = axes[1]
    ax.set_facecolor('#161b22')
    names   = [c["name"] for c in valid if "strain_energy" in c]
    strains = [c["strain_energy"] for c in valid if "strain_energy" in c]
    colors  = ['#f97316' if n == lead["name"] else '#58a6ff' for n in names]

    bars = ax.barh(names, strains, color=colors, edgecolor='#30363d')
    ax.set_xlabel('Strain Energy (kJ/mol)', color='#aaa')
    ax.set_title('Strain Energy by Candidate\n(lower = better)', color='white')
    ax.tick_params(colors='#aaa')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')
    ax.annotate('★ Lead', xy=(strains[names.index(lead["name"])],
                               names.index(lead["name"])),
                xytext=(10, 0), textcoords='offset points',
                color='#f97316', fontweight='bold')

    plt.tight_layout(pad=2)
    fig.patch.set_facecolor('#0d1117')
    plt.show()

    print(f"\n── Lead Candidate Summary ──────────────────────")
    print(f"  Name   : {lead['name']}")
    print(f"  SMILES : {lead['smiles']}")
    print(f"  MW     : {lead['mw']:.1f}  logP: {lead['logp']:.2f}  TPSA: {lead['tpsa']:.1f}")
    print(f"  Strain : {lead['strain_energy']} kJ/mol")
    print(f"  Reason : {lead['llm_reasoning']}")


# ── Main pipeline ─────────────────────────────────────────────

def run_demo(target: str = "COX-2 inhibitor for anti-inflammatory activity", n: int = 5):
    print("═" * 60)
    print("Step 1: Generating candidates via LLM...")
    print("═" * 60)
    candidates = generate_candidates(target, n=n)

    print("\n" + "═" * 60)
    print("Step 2: Validating and featurizing with RDKit...")
    print("═" * 60)
    candidates = validate_and_featurize(candidates)

    print("\n" + "═" * 60)
    print("Step 3: Estimating binding energy via OpenMM...")
    print("═" * 60)
    candidates = estimate_binding_energy(candidates)

    print("\n" + "═" * 60)
    print("Step 4: LLM selects lead candidate...")
    print("═" * 60)
    lead = select_lead(candidates)

    print("\n" + "═" * 60)
    print("Step 5: Visualizing results...")
    print("═" * 60)
    visualize_results(candidates, lead)

    return candidates, lead


if __name__ == "__main__":
    candidates, lead = run_demo()

What each step does:
StepToolOutputGenerate SMILESClaude (LLM)5 candidate molecules with rationaleValidate + featurizeRDKit3D conformers, MW/logP/TPSA/LipinskiBinding energy proxyOpenMM (GPU)Strain energy per candidateSelect leadClaude (LLM)Reasoned pick balancing all propertiesVisualizeRDKit + matplotlib2D grid + strain energy bar chart
To run in a notebook:
pythoncandidates, lead = run_demo(
    target="kinase inhibitor targeting EGFR for NSCLC",
    n=5
)
The strain energy proxy is a simplification — a real pipeline would dock against a protein PDB using AutoDock-GPU or Glide. But this runs entirely without a protein structure file, making it self-contained for a demo while the overall loop (LLM → sim → LLM → viz) is identical to production.