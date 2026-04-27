# DS 6210 — Spectral Optimizer Capstone

Course materials and starter code for the DS 6210 take-home capstone on
spectral optimization for deep learning.

## Layout

    project/
      spectral_harness/           starter code -- read its README first
        README.md                 quickstart, rules, self-audit checklist
        my_optimizer.py           student work lives here
        harness/                  fixed scaffolding (model, data, timing, seeds)
        experiments/              4 scripts: baseline, your method, parity, grid
        tests/                    pytest parity contract
        reproduce.sh              end-to-end smoke

## Quickstart

```bash
cd project/spectral_harness
pip install -r requirements.txt
./reproduce.sh
```

The full assignment sheet (problem statement, rubric, rules) is
distributed separately by the instructor. Read
[`project/spectral_harness/README.md`](project/spectral_harness/README.md)
for the starter API and the self-audit checklist.
