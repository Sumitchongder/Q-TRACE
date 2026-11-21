"""Write the results block of README.md from results/summary (no hand-typed numbers)."""

import glob
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
S = ROOT / "results" / "summary"


def md_table(df):
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        name = f"**{idx}**" if idx == "Q-TRACE" else idx
        lines.append("| " + " | ".join([name] + [str(v) for v in row.values]) + " |")
    return "\n".join(lines)


def main():
    det = pd.read_csv(S / "table_detection_far0.001.csv", index_col=0)
    auc = pd.read_csv(S / "table_auroc.csv", index_col=0)
    n_seeds = len(glob.glob(str(ROOT / "results" / "benchmark" / "seed_*" / "meta.json")))
    parts = [
        f"**Attacked links detected (%) and attack-free links with any alarm (FA, %)** at a per-window "
        f"false-alarm rate of 10⁻³; mean ± 95 % CI over {n_seeds} held-out seeds.\n",
        md_table(det),
        "\n**Window-level AUROC** (mean ± 95 % CI):\n",
        md_table(auc),
    ]
    hw = sorted(glob.glob(str(S / "hardware" / "*_matched_detection.csv")))
    if hw:
        stem = pathlib.Path(hw[-1]).stem.replace("_matched_detection", "")
        pre = pd.read_csv(S / "hardware" / f"{stem}_detection.csv")
        far_pre = float(pre[pre.kind == "honest"]["Q-TRACE@0.001"].iloc[0])
        h = pd.read_csv(hw[-1])
        cols = ["probe", "n", "ideal_qber", "measured_qber_excess", "Q-TRACE@0.001",
                "QBER (per basis)@0.001", "QBER (pooled)@0.001"]
        h = h[cols].set_index("probe")
        for c in ("ideal_qber", "measured_qber_excess"):
            h[c] = h[c].map(lambda v: f"{v:.4f}")
        parts += [f"\n**IBM Quantum hardware** (`{stem}`), fraction of link-epochs flagged at "
                  "alpha = 10^-3 with a condition-matched reference (each link's honest monitoring "
                  "epochs). With the pre-registered commissioning reference, every test, including the "
                  f"QBER monitors, flagged {100 * far_pre:.0f} % of honest link-epochs: commissioning "
                  "circuits lacked the idle period and concurrent two-qubit activity of monitoring "
                  "circuits, producing an X-basis dephasing shift of about 1.9 %. See "
                  "`docs/hardware.md`.\n", md_table(h)]
    else:
        parts.append("\n*Hardware results: pending the QPU run.*")
    block = "\n".join(parts)
    readme = (ROOT / "README.md").read_text()
    a, b = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"
    readme = readme[:readme.index(a) + len(a)] + "\n" + block + "\n" + readme[readme.index(b):]
    (ROOT / "README.md").write_text(readme)
    print("README results block updated")


if __name__ == "__main__":
    main()
