# geothermal-orc

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khb-git/geothermal-orc/blob/main/notebooks/demo.ipynb)
[![View on nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/khb-git/geothermal-orc/blob/main/notebooks/demo.ipynb)

A single, deep, literature-grade model of a **subcritical binary (Organic
Rankine Cycle) geothermal power plant** — from working-fluid selection through
cycle thermodynamics, evaporator pinch analysis, resource-coupled energy and
exergy balances, evaporation-temperature optimization, the **balance-of-plant**
that turns gross power into net-to-grid (parasitic fans and pumps, ambient-set
condensing, seasonal output), **off-design operation** as the resource cools,
and the two real-world constraints that bound binary plant design: **silica
scaling** and **reservoir thermal decline**.

Built on [CoolProp](http://www.coolprop.org/) for equation-of-state property
data, with [SciPy](https://scipy.org/) for the root-finding and optimization.

---

## Why this exists

Most teaching ORC scripts stop at a per-kilogram cycle efficiency. Real
geothermal binary design is dominated by things a per-kg cycle never sees:

- the **pinch** inside the evaporator, which sits at the working-fluid bubble
  point rather than at either terminal, so a log-mean-ΔT check is not enough;
- the **resource coupling** that ties working-fluid flow to the brine stream and
  decides how much power the field can actually support;
- the **parasitic loads and ambient heat sink** that separate turbine *gross*
  power from *net-to-grid*, and make output breathe with the seasons;
- the **off-design** behaviour of fixed installed hardware as the resource cools,
  which differs from re-optimizing a fresh plant each year;
- the **silica chemistry** that sets a floor on how cold the brine may be
  reinjected;
- the **thermal decline** that erodes output over field life.

This package models all of them, and checks itself against published numbers and
against first-law / second-law balance closure.

---

## Installation

```bash
git clone https://github.com/khb-git/geothermal-orc.git
cd geothermal-orc
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Requires Python 3.10+. Core dependencies: `CoolProp`, `numpy`, `scipy`,
`matplotlib`, `pandas`. The `[dev]` extra adds `pytest`, `jupyter`, and
`nbconvert` for the test suite and demo notebook.

---

## Quickstart

```python
from geothermal_orc import ORCCycle, GeothermalResource, optimize_evaporation_temperature

# A subcritical isobutane cycle, 120 C evaporation / 30 C condensation.
cyc = ORCCycle("Isobutane", T_evap_C=120.0, T_cond_C=30.0,
               eta_pump=0.75, eta_turbine=0.80)

# Couple it to a 150 C brine at 100 kg/s; size the plant to a 5 K pinch.
res = cyc.solve_with_resource(m_brine=100.0, T_brine_in_C=150.0, pinch_evap=5.0)
print(res.summary())
print("energy residual:", res.energy_balance_residual)   # ~1e-16
print("exergy residual:", res.exergy_balance_residual)   # ~1e-13

# Find the evaporation temperature that maximizes net power for this resource.
resource = GeothermalResource(T_reservoir_C=150.0, mass_flow=100.0)
opt = optimize_evaporation_temperature("Isobutane", resource, T_cond_C=30.0)
print(f"optimum: {opt.T_evap_opt_C:.1f} C -> {opt.W_net_opt/1e6:.2f} MW")
```

For deliverable power, the `plant` layer adds parasitics and an ambient-set heat
sink:

```python
from geothermal_orc import evaluate_plant

pr = evaluate_plant("Isobutane", resource, ambient_C=10.0)   # ~30 C condensing
print(f"gross {pr.W_gross/1e6:.2f} MW -> net-to-grid {pr.W_net_plant/1e6:.2f} MW "
      f"({pr.net_gross_ratio*100:.0f}% of gross)")
```

The full walkthrough — a ten-act story from "what is a binary plant" to a design
verdict, with the plant schematic, *T*–*s* and *T*–*Q* diagrams, an
exergy-destruction breakdown, the gross-to-net waterfall and seasonal output,
fluid screening, silica curves, a re-optimized-vs-fixed-hardware decline, and an
interactive design explorer — is in
[`notebooks/demo.ipynb`](notebooks/demo.ipynb) (executed, with plots embedded).

**Viewing it:** GitHub renders the notebook with all plots inline; if its viewer
ever hiccups (the occasional "An error occurred" page), the **nbviewer** badge
above always renders it. To move the explorer's sliders, click **Open in Colab**
and run all cells — Colab installs the package and gives you a live kernel.

---

## Module map

| module | what it does |
|--------|--------------|
| `thermo` | `State` wrapper over CoolProp; flow-exergy relative to a dead state; saturation/critical helpers. |
| `fluids` | 14-fluid library with ASHRAE-34 safety, ODP, GWP100, and a wet/dry/isentropic classifier; environmental `screen()`. |
| `heat_exchanger` | LMTD, and `counterflow_profile` — a duty-swept *T*–*Q* profile that resolves the internal pinch through phase change. |
| `cycle` | `ORCCycle`: the four-point cycle (`solve`) and the resource-coupled plant with full energy/exergy balances (`solve_with_resource`). |
| `geothermal` | Fournier–Rowe silica solubility (amorphous & quartz), the quartz geothermometer, the reinjection-temperature floor, and a `GeothermalResource` with a thermal-decline model. |
| `optimization` | `optimize_evaporation_temperature` (net-power maximization under subcritical/pinch/silica constraints) and `screen_fluids` (multi-fluid ranking). |
| `plant` | Balance of plant: parasitic loads (air-cooled-condenser fans, brine pumping), ambient-set condensing, `evaluate_plant` (gross → net-to-grid), `seasonal_performance`, and off-design operation (`design_plant`, `off_design_operation`, `decline_curves`) via a Stodola swallowing law, fixed UA, and a part-load turbine model. |

State-point numbering follows DiPippo (2016): **1** condenser outlet / pump
inlet, **2** pump outlet, **3** evaporator outlet / turbine inlet, **4** turbine
outlet / condenser inlet.

---

## Testing

```bash
pytest -q
```

The suite is **117 tests** and targets correctness rather than coverage theatre:
energy- and exergy-balance closure on resource-coupled solves, pinch behaviour
across phase change, silica-correlation values against the literature, the
classifier's wet/dry/isentropic verdicts, optimizer feasibility/ranking,
parasitic and off-design plant behaviour, and a `test_benchmarks.py` suite that
pins headline numbers to published values.

> Performance note: `solve_with_resource` precomputes the (flow-independent)
> working-fluid side of the evaporator profile once and runs the pinch root-find
> over the brine side only, which keeps the optimizer and fluid screen fast
> without changing the full-resolution result.

---

## Validation status (read this before trusting a number)

This package is honest about what is *validated* versus what is merely
*plausible*:

**Hard validations** (matched to published values):

- Amorphous-silica and quartz solubilities reproduce Fournier & Rowe (1977) to
  within a few mg/kg (e.g. amorphous ≈ 117 mg/kg at 25 °C vs. a literature
  115–120; quartz ≈ 48.1 mg/kg at 100 °C vs. 48–52).
- The quartz geothermometer round-trips: 200 °C → 265 mg/kg → 201 °C.
- Energy and exergy balances on every resource-coupled solve close to machine
  precision (residuals ~1e-16 and ~1e-13).
- Wet/dry/isentropic fluid classes agree with the standard literature consensus
  (e.g. water/ammonia/R134a wet; isobutane/pentanes/R245fa dry;
  R1234yf/R1234ze(E) near-isentropic).
- **Benchmarked against the ORC literature** (`tests/test_benchmarks.py`):
  subcritical thermal efficiency at 100 °C/30 °C lands at ~11–12% (the ~10%
  consensus of Saleh et al. 2007 and Su et al. 2018), ideal-cycle efficiency
  ~15%, and the 150 °C fluid screen tops out on isobutane/propane — matching
  Augustine et al. (NREL), who found isobutane optimal for 140–170 °C binary.

**Plausibility checks** (right physics, right ballpark — *not* matched to a
named plant):

- Absolute net-to-grid power, utilization (second-law) efficiency, and brine
  outlet temperature fall within ranges DiPippo reports for comparable ~150 °C
  binary resources, but are not calibrated to a specific operating unit.
- Parasitic-load defaults (fan air-side ΔT and Δp, brine-loop pressure) are
  documented, site-independent starting points; a real project substitutes
  measured values.
- The off-design model uses a reduced Stodola + fixed-UA + part-load-turbine
  representation; the part-load curve is illustrative, not a specific machine map.
- Decline rates use the Snyder et al. (2017) averages (~0.5 %/yr binary,
  ~0.8 %/yr flash, predominantly linear); an individual field can differ.

---

## Scope and limitations

The model is a **pure-fluid, subcritical, non-recuperated** binary cycle with a
**reduced** balance-of-plant. Deliberate simplifications, and the planned
extensions that address them:

- **No recuperator/IHX, no superheat control** beyond saturated-vapour turbine
  inlet. (For dry fluids, recuperation can *lower* net power on a free heat
  source — a geothermal-specific nuance worth modelling.)
- **Pure fluids only.** Zeotropic mixtures glide in temperature and better match
  the brine and condenser curves — the standard route to higher utilization.
- **Subcritical only.** Transcritical/supercritical cycles can match the resource
  better at the cost of higher pressures.
- **Isobaric heat exchange.** Real evaporator/condenser/piping pressure drops are
  neglected; the pinch is an input rather than an area/cost trade-off.
- **No techno-economics.** Power is the objective; LCOE and a cost-optimal (vs
  power-optimal) design are not yet modelled.
- **Brine as pure water**; calcite/CO₂/H₂S chemistry beyond silica is not modelled
  (binary's pressurized, no-flash operation does suppress calcite).
- **Empirical decline**; reservoir-scale thermal breakthrough is out of scope.

These are tracked as a tiered roadmap (parasitics and ambient/off-design are
done; mixtures, recuperation, pressure drops, and a techno-economic/LCOE layer
are next).

---

## References

- R. DiPippo, *Geothermal Power Plants: Principles, Applications, Case Studies
  and Environmental Impact*, 4th ed., Butterworth-Heinemann, 2016.
- B. Saleh, G. Koglbauer, M. Wendland, J. Fischer, "Working fluids for
  low-temperature organic Rankine cycles," *Energy* 32 (2007) 1210–1221.
- R. O. Fournier, J. J. Rowe, "The solubility of amorphous silica in water at
  high temperatures and high pressures," *American Mineralogist* 62 (1977)
  1052–1056.
- D. M. Snyder, K. F. Beckers, K. R. Young, "Update on geothermal direct-use and
  power-plant thermal decline," *GRC Transactions* 41 (2017).
- C. Augustine et al., "A comparison of geothermal binary cycle working fluids"
  (NREL), *GRC Transactions* — isobutane optimal for 140–170 °C subcritical
  binary resources.
- W. Su et al., "A limiting efficiency of subcritical Organic Rankine cycle under
  the constraint of working fluids," *Energy* (2018) — subcritical ORC thermal
  efficiency ~10%, below 50% of Carnot perfectness.

---

## License

MIT — see [LICENSE](LICENSE).
