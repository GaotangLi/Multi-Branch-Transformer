# Figure assets

The README figures use the current paper's palette and Avenir/Chalkboard
typography. PNGs are **4,500 pixels wide**; matching SVGs retain vector geometry
and outlined text, so fonts do not need to be installed. Click a homepage figure
to open its vector version.

| Asset | Content |
|:---|:---|
| `overview.png` / `.svg` | Latest paper architecture comparison and performance–speed frontier. |
| `width-depth.png` / `.svg` | Latest paper Wide–Shallow, Narrow–Deep, and Balanced allocations. |
| `final-model-scaling.png` / `.svg` | Standard and final Balanced Joint-KV models, with fitted scaling laws. |
| `final-model-results.png` / `.svg` | Final-model scaling above the paper's mean decode-speedup panels. |

The architecture and allocation figures are exported directly from the paper's
vector PDFs. The scaling panels use the same axes, points, fits, and style,
showing only Standard and the final Joint-KV models. The decode panels are taken
directly from the paper. Numerical results are unchanged.

- `final-model-losses.csv`: full-precision validation losses.
- `final-model-scaling.json`: fitted coefficients and equal-loss compute comparisons.
- `mbt-wordmark.svg` / `mbt-icon.svg`: editable repository wordmark and icon.
- `joint-kv.png` / `.svg`: supplementary KV-sharing illustration (3,000-pixel PNG).
- `badges/*.svg`: publication and resource badges.

These are presentation assets, not checkpoints or training data. Font files are
not distributed. See the repository license and third-party notices for reuse.
