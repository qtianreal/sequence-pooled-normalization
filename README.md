# Sequence-pooled normalization and the receptive field

Code, simulated data generators, and result files for the study of how a
normalization layer whose statistics are pooled along the sequence supplies
context that the convolutional receptive field is usually credited with.

A normalization layer that computes its statistics from the current input,
along the sequence, at inference gives every output position a summary of the
whole input, through a path that no receptive-field calculation accounts for
because it passes through no convolution. Where labels come in long runs, that
summary supplies most of the usable context: a network reaching 9 positions
comes within 0.009 of the whole-sequence optimum, while the best any predictor
confined to those 9 positions could do is 0.548. Closing the path, by taking
the same statistics per position, multiplies what enlarging the receptive field
is worth by up to an order of magnitude, and the same path inflates what block
ablation attributes to a network's reach-enlarging blocks.

The manuscript is under review.

## Requirements

```
pip install -r requirements.txt
```

Python 3.10 or newer. The experiments were run on a single consumer GPU; the
training scripts select Apple `mps` where available and fall back to CPU.
`msprime` is needed only for the simulated genomic process, and
`scikit-allel` only for the real-haplotype arm.

## Layout

```
difficulty/           the library: processes, models, bounds, ablation
  axis/synthetic.py     Gaussian--Markov process
  axis/genomic.py       coalescent ancestry process
  models.py             dilated stack, U-Net, transformer, Conv-TasNet separator,
                        and the normalization variants
  task.py               bounds and exact optima (forward--backward)
  measure.py            Jacobians, receptive fields by autograd, statistics
  ablation.py           block removal and retraining
  real.py               1000 Genomes mosaics
run_*.py              one script per experiment; each writes results/*.json
results/*.json        every number in the paper comes from these
paper/make_*.py       turn results/*.json into the tables, macros and figures
paper/numbers.tex     generated: every number quoted in the prose
paper/tables.tex      generated: the appendix tables
paper/figures/        generated figures
data/kg.panel         1000 Genomes sample-to-population map (public metadata)
```

## Reproducing the paper's numbers

The pipeline has two stages. The experiments write JSON to `results/`; the
generators read `results/` and write every table, macro and figure. Run from
the repository root.

```
python run_optima.py            # exact optima and bounds (no training)
python run_tradeoff.py          # difficulty sweep, ablation and retraining
python run_synth.py             # synthetic process
python run_tracts.py            # simulated genomic process
python run_real_norm.py         # real 1000 Genomes haplotypes
python run_unet.py              # U-Net depth sweep
python run_jacobian.py          # numerical check of the Jacobian identity
python run_port_check.py        # F_ST against the benchmark ported from
python run_signal_check.py      # pooled statistic against its closed form
python run_null.py              # label null for the discarded deff analysis
python run_shift_null.py        # the circular-shift null that failed
python run_arch.py              # additional architecture arms

python paper/make_numbers.py      # -> paper/numbers.tex, tables.tex, table_dissect.tex
python paper/make_figures.py      # -> paper/figures/fig0..fig3
python paper/make_fig_decomp.py   # -> paper/figures/fig4_decomp
python paper/make_fig_criterion.py # -> paper/figures/exposure_criterion
```

The generated files are committed, so a regeneration can be diffed against the
version the manuscript was built from. `make_numbers.py` fails loudly rather
than silently if a result file is missing or if a claim it asserts (for
example, that a quoted value is the extremum of its sweep) no longer holds.

### Which script regenerates which table

Tables are named by their LaTeX label rather than by number, since numbering
depends on the manuscript build.

| Table | Reads | Produced by |
| --- | --- | --- |
| `tab:axes` — reach worth by pooling axes | `tracts_norms.json` | `run_tracts.py` (normalization sweep) |
| `tab:dissect` — ablation against retraining by pooling axes | `tracts_norms.json` | `run_tracts.py` (normalization sweep) |
| `tab:grid` — accuracy by reach and divergence | `tradeoff.json` | `run_tradeoff.py` |
| `tab:difficulty` — reach worth per divergence level | `tradeoff.json` | `run_tradeoff.py` |
| `tab:ablcost` — ablation and retraining costs per level | `tradeoff.json` | `run_tradeoff.py` |
| `tab:dose` — dose--response in switch rate | `synth_dilated_cnn.json`, `tracts_generations.json`, `real_norm.json` | `run_synth.py`, `run_tracts.py`, `run_real_norm.py` |
| `tab:tasnet` — Conv-TasNet curves | `synth_tasnet.json` | `run_synth.py --arch tasnet` |
| `tab:transformer` — transformer control | `synth_transformer.json` | `run_synth.py --arch transformer` |
| `tab:resid` — networks against the local bound, cell by cell | `optima.json`, `synth_dilated_cnn.json` | `run_optima.py`, `run_synth.py` |
| `tab:optima` — local bounds and exact optima | `optima.json` | `run_optima.py` |
| `tab:deriv` — closed form against the computed optima | `optima.json` | `run_optima.py` |
| `tab:port` — simulated F_ST against the ported benchmark | `port_check.json` | `run_port_check.py` |
| `tab:unet` — U-Net depth sweep | `unet.json` | `run_unet.py` |
| `tab:survey` — normalization survey of published models | `exposure_survey.json` | curated by reading each project's source; see below |

### Which script regenerates which figure

| Figure | Reads | Produced by |
| --- | --- | --- |
| `fig0_mechanism` — the two paths, schematic | — | `paper/make_figures.py` |
| `fig1_reach` — accuracy against receptive field | `tradeoff.json`, `real_norm.json`, `synth_tasnet.json` | `paper/make_figures.py` |
| `fig2_dose` — dose--response | `synth_dilated_cnn.json`, `tracts_generations.json` | `paper/make_figures.py` |
| `fig3_ablation` — ablation against retraining | `tracts_norms.json`, `tradeoff.json`, `real_norm.json` | `paper/make_figures.py` |
| `fig4_decomp` — the information landscape | `optima.json`, `synth_dilated_cnn.json` | `paper/make_fig_decomp.py` |
| `exposure_criterion` — the criterion at the layer | `exposure_criterion_overview.svg` | `paper/make_fig_criterion.py` |

### Result files whose names differ from the script that writes them

Three released result files come from running one script more than once with
different arguments, saved under descriptive names rather than the name the
script writes:

- `run_tracts.py` writes `results/tracts.json`. The released
  `tracts_generations.json` is the run that sweeps `--generations` (the
  dose--response on the genomic process); `tracts_norms.json` is the run that
  sweeps `--norms` over all five normalizations at a fixed generation count.
- `run_synth.py` writes `results/synth_<arch>.json`. The released
  `synth_dilated_cnn.json` is the dense switch-rate sweep;
  `synth.json` is the same architecture run at the four switch densities the
  genomic process happens to produce, so the two processes can be compared
  point for point.

Rename the output accordingly, or pass the file name your own analysis expects.

### The survey file

`results/exposure_survey.json` is not produced by a script. Each row records a
reading of a published model's released source, pinned to the commit that was
read, with the normalization configuration found there and whether it meets the
criterion. Models examined and excluded are recorded with their reasons.

## Data not redistributed here

The real-haplotype arm needs 1000 Genomes chromosome 22 (phase 3), which is
public but not redistributed in this repository. `data/kg.panel` is the
sample-to-population map only. Point `run_real_norm.py` at a local copy of the
VCF; the mosaics, features and splits are built by the same code as the
simulated process.

Trained weights are not included either. `run_tradeoff.py` writes them to
`results/weights/` as it trains, which is what `run_null.py` reads.

## License

MIT; see `LICENSE`. The 1000 Genomes data referenced by the real-haplotype arm
is distributed by its own project under its own terms.

## Notes

- Every number in the manuscript is generated: the prose contains no
  hand-typed figures, and `paper/make_numbers.py` is the single source for all
  of them.
- Uncertainties reported are standard deviations across three seeds of the
  per-seed value of the quantity in question. Compared models are paired on
  data and seed.
- The experiments total 729 trained-and-evaluated configurations.
