# geothermal-orc

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khb-git/geothermal-orc/blob/main/notebooks/demo.ipynb)
[![View on nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/khb-git/geothermal-orc/blob/main/notebooks/demo.ipynb)

A single, deep, literature-grade model of a **subcritical binary (Organic
Rankine Cycle) geothermal power plant** — from working-fluid selection through
cycle thermodynamics, evaporator pinch analysis, resource-coupled energy and
exergy balances, evaporation-temperature optimization, the **balance-of-plant**
that turns gross power into net-to-grid (parasitic fans and pumps, ambient-set
condensing, seasonal output), **off-design operation** as the resource cools,
**cycle-fidelity layers** (recuperation, pressure drops, the pinch/area trade-off)
and **advanced configurations** (zeotropic mixtures with temperature glide, and
transcritical cycles), and the two real-world constraints that bound binary
plant design: **silica scaling** and **reservoir thermal decline**.

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

And the `economics` layer turns net power into a cost of electricity:

```python
from geothermal_orc import levelized_cost

econ = levelized_cost("Isobutane", resource, ambient_C=10.0, T_evap_C=95.0)
print(f"LCOE {econ.lcoe:.0f} USD/MWh  |  {econ.specific_capex:,.0f} USD/kW  "
      f"|  wells {econ.capex_wells/econ.capex_total*100:.0f}% of CAPEX")
```

…and the `finance` layer turns that cost into an investment decision:

```python
from geothermal_orc import project_cashflow, ProjectAssumptions, monte_carlo

pr = project_cashflow(econ, ProjectAssumptions(ppa_price=120.0))
print(f"NPV ${pr.npv/1e6:.1f}M | IRR {pr.irr*100:.0f}% | payback {pr.payback_yr:.0f} yr")
mc = monte_carlo(econ, ProjectAssumptions(ppa_price=110.0))
print(f"P(NPV>0) = {mc['p_npv_positive']*100:.0f}%")
```

The full walkthrough — a fourteen-act story from "what is a binary plant" to a
design verdict, advanced cycles, the cost of a megawatt-hour, and whether to
build it, with the plant schematic, *T*–*s* and *T*–*Q* diagrams, an
exergy-destruction breakdown, the gross-to-net waterfall and seasonal output,
fluid screening, silica curves, a re-optimized-vs-fixed-hardware decline, the
recuperator/pressure-drop/pinch-area refinements, the zeotropic and transcritical
advanced cycles, the CAPEX breakdown and cost-optimal-vs-power-optimal pinch, and
the NPV/IRR and Monte-Carlo risk analysis, plus an interactive design explorer —
is in
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
| `cycle` | `ORCCycle`: the four-point cycle (`solve`) and the resource-coupled plant with full energy/exergy balances (`solve_with_resource`); optional recuperator and heat-exchanger pressure drops. |
| `geothermal` | Fournier–Rowe silica solubility (amorphous & quartz), the quartz geothermometer, the reinjection-temperature floor, and a `GeothermalResource` with a thermal-decline model. |
| `optimization` | `optimize_evaporation_temperature` (net-power maximization under subcritical/pinch/silica constraints) and `screen_fluids` (multi-fluid ranking). |
| `plant` | Balance of plant: parasitic loads (air-cooled-condenser fans, brine pumping), ambient-set condensing, `evaluate_plant` (gross → net-to-grid), `seasonal_performance`, off-design operation (`design_plant`, `off_design_operation`, `decline_curves`), and the `pinch_area_tradeoff`. |
| `mixtures` | Zeotropic working-fluid mixtures with temperature glide: `MixtureCycle` (built on P-T/P-Q flashes since CoolProp lacks mixture P-H/P-S), glide helpers, and `screen_compositions`. |
| `transcritical` | `TranscriticalCycle`: supercritical heat addition (no evaporation plateau) with subcritical condensing, for resources whose hot end a subcritical fluid cannot reach. |
| `economics` | Techno-economics: Turton module costing of every component (CEPCI-updated), geothermal well costs, and `levelized_cost` → LCOE, plus `pinch_lcoe_tradeoff` (cost-optimal vs power-optimal design) and `lcoe_sensitivity`. |
| `finance` | Project finance and risk: a discounted-cash-flow model (`project_cashflow` → NPV, IRR, payback, break-even PPA) with optional debt, tax, and depreciation, and a `monte_carlo` layer that turns the point LCOE into a distribution and a probability of positive NPV. |

State-point numbering follows DiPippo (2016): **1** condenser outlet / pump
inlet, **2** pump outlet, **3** evaporator outlet / turbine inlet, **4** turbine
outlet / condenser inlet.

---

## Testing

```bash
pytest -q
```

The suite is **156 tests** and targets correctness rather than coverage theatre:
energy- and exergy-balance closure on resource-coupled solves, pinch behaviour
across phase change, silica-correlation values against the literature, the
classifier's wet/dry/isentropic verdicts, optimizer feasibility/ranking,
parasitic and off-design plant behaviour, recuperator/pressure-drop/mixture/
transcritical physics, techno-economic LCOE in the published range with the
right cost structure, the break-even-equals-LCOE finance identity, and a
`test_benchmarks.py` suite that pins headline numbers to published values.

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
- **Advanced cycles preserve balance closure and match the literature
  direction:** recuperated, pressure-drop, mixture, and transcritical solves all
  keep energy/exergy balances closed; zeotropic mixtures help low-enthalpy
  resources (~5% at 120 °C, per Heberle/Chys) and transcritical propane helps at
  150 °C (~5%, per Astolfi), while neither beats a well-matched pure fluid where
  one already fits.
- **Techno-economics in range:** Turton module costing plus geothermal wells
  gives ~$90/MWh and ~$6,700/kW for the 150 °C base case (inside the published
  band for medium-temperature binary; Lazard/IRENA), with wells the dominant
  (~60%+) share of CAPEX.
- **Finance reproduces the LCOE:** under all-equity, no-tax, matching-discount
  assumptions the break-even PPA equals the Tier 3 LCOE to within rounding and
  the IRR at that price equals the discount rate — the cash-flow model is
  internally consistent with the cost model before debt, tax, escalation, and
  Monte-Carlo risk are layered on.

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

The core cycle now spans subcritical and transcritical operation, pure fluids
and zeotropic mixtures, with an optional recuperator, heat-exchanger pressure
drops, a reduced balance-of-plant, off-design behaviour, a techno-economic LCOE
layer, and a project-finance/risk layer. What it **models**:

- Subcritical and **transcritical** cycles; **pure fluids and zeotropic
  mixtures** (temperature glide); optional **recuperator**; heat-exchanger
  **pressure drops**; the **pinch/area** (UA) trade-off.
- Parasitics and ambient-set condensing (gross → net-to-grid), seasonal output,
  and fixed-hardware **off-design** decline.
- **Techno-economics:** component sizing and Turton module costing, geothermal
  well costs, **LCOE**, and the **cost-optimal vs power-optimal** design question
  (how much heat-exchanger area is worth buying).
- **Project finance & risk:** discounted cash flow with optional debt, tax and
  depreciation → **NPV, IRR, payback, break-even PPA**, and a **Monte-Carlo**
  distribution of LCOE/NPV with the probability of a positive return.

Deliberate simplifications that remain:

- **First-order costing.** Turton module costing with material/pressure factors
  held at a carbon-steel, low-pressure baseline; well count, well cost, O&M, and
  finance terms are documented, adjustable assumptions, not a bankable model.
- **Monte-Carlo holds the plant design fixed.** Uncertainty is propagated through
  the economic/financial drivers (PPA, well cost, capacity factor, cost of
  capital); resource-temperature uncertainty enters only through the
  deterministic sensitivities, not a re-solved thermodynamic sample.
- **Reduced off-design.** A Stodola swallowing law plus fixed UA and an
  illustrative part-load turbine curve, not a manufacturer's component map.
- **Mixture comparison anchored on mean condensing temperature** (a fair proxy);
  a full cooling-water-matched condenser with floating pressure is not yet built.
- **Brine as pure water**; calcite/CO₂/H₂S chemistry beyond silica is not modelled
  (binary's pressurized, no-flash operation does suppress calcite).
- **Empirical decline**; reservoir-scale thermal breakthrough is out of scope.

Roadmap status: parasitics, ambient/seasonal, and off-design (Tier 1), the
fidelity layer — recuperation, pressure drops, mixtures, transcritical (Tier 2) —
the techno-economic LCOE layer (Tier 3), and the project-finance/risk layer
(Tier 4) are **done**; a fully cooling-water-matched condenser with floating
condensing pressure, and Monte-Carlo over a re-solved thermodynamic model
(resource-temperature uncertainty), are the natural next steps.

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
- F. Heberle, D. Brüggemann, "Thermo-economic evaluation of organic Rankine
  cycles for geothermal power generation using zeotropic mixtures," *Energies*
  (2015) — glide-matching raises second-law efficiency for low-enthalpy
  resources.
- C. Augustine et al., "A comparison of geothermal binary cycle working fluids"
  (NREL), *GRC Transactions* — isobutane optimal for 140–170 °C subcritical
  binary resources.
- W. Su et al., "A limiting efficiency of subcritical Organic Rankine cycle under
  the constraint of working fluids," *Energy* (2018) — subcritical ORC thermal
  efficiency ~10%, below 50% of Carnot perfectness.
- R. Turton, R. C. Bailie, W. B. Whiting, J. A. Shaeiwitz, *Analysis, Synthesis
  and Design of Chemical Processes*, Prentice Hall — module-costing method and
  equipment cost correlations (USD 2001 basis, updated via CEPCI).
- F. Yang et al., "Thermo-economic optimization of organic Rankine cycle systems
  for geothermal power generation," *Front. Energy Res.* 8 (2020) 6 — finds
  recuperation economically unattractive (extra exchanger cost outweighs the
  gain) and transcritical cycles favoured, validating the Tier 2/3 findings.
- Lazard, *Levelized Cost of Energy+*, and IRENA, *Renewable Power Generation
  Costs* — geothermal LCOE benchmark (~$60–150/MWh for medium-temperature binary).
- W. Short, D. J. Packey, T. Holt, *A Manual for the Economic Evaluation of
  Energy Efficiency and Renewable Energy Technologies*, NREL/TP-462-5173 (1995) —
  LCOE, capital-recovery factor, and discounted-cash-flow conventions.

---

## License

MIT — see [LICENSE](LICENSE).
