"""Generate notebooks/demo.ipynb programmatically with nbformat.

Run from the repo root:  python notebooks/_build_demo.py
Then execute:            jupyter nbconvert --to notebook --execute --inplace notebooks/demo.ipynb
Then finalize:           python notebooks/_finalize_demo.py
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []


def md(text):
    cells.append(new_markdown_cell(text))


def code(src):
    cells.append(new_code_cell(src))


def takeaway(text):
    cells.append(new_markdown_cell("> **Takeaway —** " + text))


# ============================================================================ #
md(r"""# From a Hot Well to Watts

### Designing a subcritical binary geothermal power plant, end to end

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khb-git/geothermal-orc/blob/main/notebooks/demo.ipynb)
&nbsp;
[![View on nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/khb-git/geothermal-orc/blob/main/notebooks/demo.ipynb)

**The question this notebook answers.** You are handed one geothermal well. It
produces **100 kg/s of 150 °C brine**. How much electricity can you pull from
it — with which working fluid, at what design point, and how does that hold up
over thirty years of production?

We will build the answer one layer at a time, using the
[`geothermal-orc`](https://github.com/khb-git/geothermal-orc) package. Each
section opens with a plain question and answers it in words; the equations,
correlations, and validation numbers sit alongside for readers who want to check
the engineering. No mathematics is required to follow the story.

**Literature anchors.** Cycle conventions follow DiPippo, *Geothermal Power
Plants* (2016); working-fluid screening follows Saleh et al. (2007); silica
solubility uses Fournier & Rowe (1977); thermal-decline rates follow Snyder et
al. (2017). Computed numbers are cross-checked against published values, and the
notebook is explicit about which checks are hard validations versus plausibility
checks.""")

code(r"""# Setup. On Colab / Binder this installs the package; locally it is a no-op.
try:
    import geothermal_orc  # noqa: F401
except ImportError:
    import subprocess, sys
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "git+https://github.com/khb-git/geothermal-orc.git"],
        check=True,
    )""")

code(r"""import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

import geothermal_orc as g
from geothermal_orc import (
    LIBRARY, get_fluid, screen, ORCCycle, counterflow_profile,
    GeothermalResource, optimize_evaporation_temperature, screen_fluids,
    amorphous_silica_solubility, quartz_solubility, quartz_geothermometer,
)

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 11, "axes.grid": True,
    "grid.alpha": 0.3, "axes.axisbelow": True,
})
print("geothermal-orc version:", g.__version__)
print("fluids in library:", len(LIBRARY))""")

# ============================================================================ #
# ACT 1
md(r"""## Act 1 · A hot well, and a problem

A geothermal resource is, in the end, a stream of hot water. Drill into the
right formation and you get brine at some temperature the geology decided for
you — here, 150 °C. That is the whole fuel supply: you cannot turn it up, and
when you are done extracting heat you must pump the brine back underground to
keep the reservoir pressured and the dissolved minerals contained.

At 150 °C the brine is too cool to make much steam, so a **flash** plant (boil
the brine itself and run the steam through a turbine) is a poor fit. The answer
is a **binary** plant. The brine never touches the turbine. Instead it heats a
*second* fluid — an organic working fluid with a low boiling point — in a heat
exchanger. That organic fluid boils, drives the turbine, condenses, and is
pumped back to do it again, in a sealed loop. This is an **Organic Rankine
Cycle (ORC)**: same idea as the steam Rankine cycle in a coal plant, but with a
fluid chosen to boil at the temperatures a moderate geothermal resource can
reach.

The schematic below shows the two loops. The **brine loop** (red) runs from the
production well, through the evaporator, to the reinjection well. The closed
**ORC loop** (teal) carries the organic fluid through its four states — pump,
evaporator, turbine, condenser — the same numbering we use everywhere
below.""")

code(r"""from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

RED, TEAL, INK, STEEL = "#8b3a1f", "#2a7d6f", "#1a1f2b", "#457b9d"

fig, ax = plt.subplots(figsize=(9.6, 5.4))
ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off"); ax.set_aspect("equal")


def box(x, y, w, h, label, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 linewidth=1.6, edgecolor=ec, facecolor="white", zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=10.5, color=INK, zorder=3)


def arrow(p0, p1, color, lw=2.3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16,
                 lw=lw, color=color, zorder=1))


def badge(x, y, n):
    ax.add_patch(Circle((x, y), 0.24, color=INK, zorder=4))
    ax.text(x, y, str(n), color="white", ha="center", va="center",
            fontsize=9.5, zorder=5, fontweight="bold")


box(0.4, 4.3, 2.0, 1.0, "Production\nwell", RED)
box(0.4, 1.4, 2.0, 1.0, "Reinjection\nwell", RED)
box(4.2, 3.0, 2.2, 1.6, "Evaporator", TEAL)
box(7.6, 4.7, 2.6, 1.1, "Turbine +\nGenerator", TEAL)
box(7.9, 1.4, 2.3, 1.3, "Condenser", TEAL)
box(4.4, 1.4, 1.8, 1.0, "Feed pump", TEAL)

# brine loop (red)
arrow((2.4, 4.8), (4.2, 4.25), RED)
arrow((4.2, 3.25), (2.4, 1.95), RED)
ax.text(2.55, 4.95, "150 °C brine", color=RED, fontsize=9)
ax.text(2.75, 2.45, "reinjected", color=RED, fontsize=9)

# ORC loop (teal) with DiPippo state numbers
arrow((6.4, 4.35), (7.6, 5.0), TEAL)    # evap -> turbine (3, vapour)
arrow((9.1, 4.7), (9.1, 2.72), TEAL)    # turbine -> condenser (4)
arrow((7.9, 2.0), (6.25, 1.9), TEAL)    # condenser -> pump (1)
arrow((5.3, 2.4), (5.3, 2.98), TEAL)    # pump -> evaporator (2)
badge(7.0, 4.78, 3); badge(9.1, 3.6, 4); badge(7.05, 1.78, 1); badge(5.3, 2.72, 2)

# outputs
arrow((10.2, 5.25), (11.5, 5.25), STEEL)
ax.text(10.35, 5.55, "electricity", color=STEEL, fontsize=9)
arrow((9.05, 1.4), (9.05, 0.5), STEEL)
ax.text(9.25, 0.75, "to cooling", color=STEEL, fontsize=9)

ax.set_title("Binary geothermal plant — brine loop (red) drives a closed ORC loop (teal)",
             fontsize=12, color=INK, pad=6)
plt.tight_layout(); plt.show()""")

takeaway(r"""A geothermal plant must live within its resource: the brine is
finite, must be reinjected, and its temperature is fixed by geology — not chosen
by the engineer. Everything that follows is the consequence of that one
constraint.""")

# ============================================================================ #
# ACT 2
md(r"""## Act 2 · Which fluid should carry the heat?

The first design lever is the organic fluid itself. Three things matter.

First, the **shape of its saturation curve**. Plot temperature against entropy
and every fluid traces a dome; what matters is the slope of the vapour side. A
*wet* fluid (like water or ammonia) turns partly back to liquid as it expands
through the turbine — droplets that erode the blades. A *dry* or *isentropic*
fluid leaves the turbine as superheated vapour, dry and safe. Binary plants want
dry or isentropic fluids. The package classifies each fluid by measuring the
slope d*s*/d*T* along the saturated-vapour line near 0.7·*T*c.

Second, **environment and safety**: ozone-depletion potential (ODP), 100-year
global-warming potential (GWP), and the ASHRAE-34 safety class (flammability and
toxicity).

Third, the **temperature fit** — the critical temperature has to sit sensibly
above the resource so the fluid can boil subcritically.

Here is the library the package ships with.""")

code(r"""import pandas as pd

rows = []
for f in sorted(LIBRARY.values(), key=lambda x: x.Tcrit):
    rows.append({
        "fluid": f.display_name, "ASHRAE": f.ashrae, "family": f.family,
        "Tcrit [C]": round(f.Tcrit - 273.15, 1),
        "Tnbp [C]": round(f.Tnbp - 273.15, 1),
        "type": f.slope_type, "safety": f.safety,
        "ODP": f.odp, "GWP100": f.gwp100,
    })
df = pd.DataFrame(rows)
df""")

md(r"""**Screening.** A modern screen keeps fluids that are not strongly *wet*,
have zero ODP, and modest GWP. Below we keep dry / isentropic fluids with GWP100
under 150 — the threshold often cited for new equipment under F-gas rules.""")

code(r"""kept = screen(max_gwp=150, max_odp=0.0, slope_types=["dry", "isentropic"])
print("passed screen:", [f.display_name for f in kept])

fig, ax = plt.subplots(figsize=(8, 4))
names = [f.display_name for f in sorted(LIBRARY.values(), key=lambda x: x.gwp100)]
gwps = [get_fluid(n).gwp100 for n in names]
kept_names = {f.display_name for f in kept}
colors = ["#2a9d8f" if n in kept_names else "#bbbbbb" for n in names]
ax.bar(names, gwps, color=colors)
ax.axhline(150, color="#e76f51", ls="--", lw=1, label="GWP100 = 150 threshold")
ax.set_ylabel("GWP$_{100}$"); ax.set_yscale("log")
ax.set_title("Working-fluid GWP (green = passes screen)")
plt.xticks(rotation=60, ha="right"); ax.legend(); plt.tight_layout(); plt.show()""")

takeaway(r"""Dry fluids that expand to superheated vapour protect the turbine.
**Isobutane** is the classic binary workhorse and our protagonist from here on —
but the screen also clears the low-GWP HFOs, and we will let the numbers pick the
winner later.""")

# ============================================================================ #
# ACT 3
md(r"""## Act 3 · The cycle — and why efficiency is not the point

With a fluid chosen, draw the cycle. Following DiPippo's convention, the working
fluid passes through four states:

| point | location | condition |
|------|----------|-----------|
| 1 | condenser outlet / pump inlet | saturated liquid at $P_\text{cond}$ |
| 2 | pump outlet | compressed liquid at $P_\text{evap}$ |
| 3 | evaporator outlet / turbine inlet | saturated (or superheated) vapour |
| 4 | turbine outlet / condenser inlet | two-phase or superheated at $P_\text{cond}$ |

We run isobutane evaporating at 120 °C and condensing at 30 °C, with realistic
pump and turbine efficiencies. The solver reports the per-kilogram work and a
thermal efficiency — and, importantly, the turbine-exit quality, which confirms
the expansion ends dry.""")

code(r"""cyc = ORCCycle("Isobutane", T_evap_C=120.0, T_cond_C=30.0,
               eta_pump=0.75, eta_turbine=0.80)
res = cyc.solve()
print(res.summary())
print(f"\nturbine exit quality Q4 = {res.turbine_exit_quality:.3f} "
      f"(-1 means superheated -> 'dry' expansion, no droplets)")""")

md(r"""The *T*–*s* diagram below traces the actual process paths, not just
straight lines between the state points: the **2 → 3** leg heats the liquid and
then runs flat across the top of the dome as the fluid boils at 120 °C, and the
**4 → 1** leg desuperheats and then condenses flat at 30 °C. Point 4 sits
*outside* the dome — the dry-expansion signature from Act 2.""")

code(r"""# Saturation dome + cycle path on T-s axes.
fluid = "Isobutane"
Tt = PropsSI("Ttriple", fluid); Tc = PropsSI("Tcrit", fluid)
Ts = np.linspace(Tt + 0.5, Tc - 0.5, 200)
s_liq = [PropsSI("S", "T", T, "Q", 0, fluid) for T in Ts]
s_vap = [PropsSI("S", "T", T, "Q", 1, fluid) for T in Ts]

st = res.states
P_evap, P_cond = st[2].P, st[1].P


def _isobar(P, h0, h1, n=80):
    # Trace an isobaric leg by sampling enthalpy: gives liquid-heating +
    # flat evaporation (or desuperheat + flat condensation) across the dome.
    hs = np.linspace(h0, h1, n)
    T = np.array([PropsSI("T", "P", P, "H", float(h), fluid) for h in hs])
    s = np.array([PropsSI("S", "P", P, "H", float(h), fluid) for h in hs])
    return s, T


# 2 -> 3 isobaric heating at P_evap; 4 -> 1 isobaric cooling at P_cond.
s23, T23 = _isobar(P_evap, st[2].h, st[3].h)
s41, T41 = _isobar(P_cond, st[4].h, st[1].h)

# Full loop: 1 --(pump)--> 2 --(evaporate)--> 3 --(turbine)--> 4 --(condense)--> 1
# (pump 1->2 and turbine 3->4 are drawn as straight segments).
s_path = np.concatenate([[st[1].s], s23, [st[4].s], s41]) / 1e3
T_path = np.concatenate([[st[1].T], T23, [st[4].T], T41]) - 273.15

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(np.array(s_liq) / 1e3, Ts - 273.15, color="#264653", lw=1.5)
ax.plot(np.array(s_vap) / 1e3, Ts - 273.15, color="#264653", lw=1.5,
        label="saturation dome")
ax.plot(s_path, T_path, color="#e76f51", lw=2, label="ORC cycle")
sm = np.array([st[i].s for i in (1, 2, 3, 4)]) / 1e3
Tm = np.array([st[i].T for i in (1, 2, 3, 4)]) - 273.15
ax.plot(sm, Tm, "o", color="#e76f51", ms=6)
for i in (1, 2, 3, 4):
    ax.annotate(str(i), (st[i].s / 1e3, st[i].T - 273.15),
                textcoords="offset points", xytext=(8, 6), fontweight="bold")
ax.set_xlabel("specific entropy  [kJ/kg·K]")
ax.set_ylabel("temperature  [°C]")
ax.set_title(f"{fluid} ORC — T–s diagram (120 °C / 30 °C)")
ax.legend(); plt.tight_layout(); plt.show()""")

takeaway(r"""Cycle efficiency measures *quality per kilogram*. It does not, by
itself, tell you megawatts — for that the cycle has to meet the resource, which
is the next move.""")

# ============================================================================ #
# ACT 4
md(r"""## Act 4 · Where the heat actually moves — the pinch

Here is the question that trips up a first design: *how much organic fluid can a
given brine stream boil?* The naive answer uses a single average temperature
difference. It is wrong, and the reason is the evaporator.

As the organic fluid reaches its boiling point its temperature stops rising — it
plateaus while it absorbs latent heat — even though the brine keeps cooling
along the exchanger. So the smallest gap between the two streams, the **pinch**,
sits *inside* the exchanger, at the fluid's bubble point, not at either end. Size
the plant on an end-to-end average and you would silently violate the pinch and
overstate the power.

`counterflow_profile` sweeps the exchanger by duty to find the pinch correctly.
Below we couple the cycle to our 150 °C / 100 kg/s brine and size the
working-fluid flow so the pinch is exactly 5 K.""")

code(r"""full = cyc.solve_with_resource(m_brine=100.0, T_brine_in_C=150.0,
                               pinch_evap=5.0)
ev = full.evaporator
print(f"working-fluid flow m_wf = {full.m_wf:.2f} kg/s")
print(f"evaporator duty        = {full.Q_in/1e6:.2f} MW")
print(f"pinch                  = {ev.pinch:.2f} K  (target 5 K)")
print(f"brine outlet           = {full.brine_T_out-273.15:.2f} °C")

fig, ax = plt.subplots(figsize=(7, 5))
Qcum = ev.duty / 1e6
ax.plot(Qcum, ev.T_hot - 273.15, color="#e76f51", lw=2, label="brine (hot)")
ax.plot(Qcum, ev.T_cold - 273.15, color="#2a9d8f", lw=2,
        label="isobutane (cold)")
i_p = int(np.argmin(ev.T_hot - ev.T_cold))
ax.annotate(f"pinch ≈ {ev.pinch:.1f} K",
            (Qcum[i_p], ev.T_cold[i_p] - 273.15),
            textcoords="offset points", xytext=(10, -25),
            arrowprops=dict(arrowstyle="->"))
ax.set_xlabel("cumulative duty  [MW]"); ax.set_ylabel("temperature  [°C]")
ax.set_title("Evaporator T–Q profile (pinch sits at the bubble point)")
ax.legend(); plt.tight_layout(); plt.show()""")

takeaway(r"""The pinch, not the cycle, sets how much working fluid the brine can
boil — and therefore the power. It is the hinge between a textbook cycle and a
real plant.""")

# ============================================================================ #
# ACT 5
md(r"""## Act 5 · So, how many megawatts?

Now the plant is sized to the brine, so we can finally answer the opening
question for a single design — and check our own arithmetic. The solver assembles
full **rate** balances and verifies two things:

* **Energy:** $Q_\text{in} - Q_\text{out} - W_\text{net} = 0$
* **Exergy:** $\dot E_\text{in} = W_\text{net} + \dot E_\text{out} +
  \sum \dot E_\text{dest}$, with each component's destruction computed as
  $T_0\,\dot S_\text{gen}$.

The **utilization efficiency** is the one a geothermal engineer cares about: not
heat-to-work, but how much of the brine's *available work* (its exergy relative
to the surroundings) the plant actually captures. Both residuals closing to
machine precision is the correctness guarantee the test-suite enforces.""")

code(r"""print(f"W_net            = {full.W_net/1e6:7.3f} MW")
print(f"Q_in             = {full.Q_in/1e6:7.3f} MW")
print(f"Q_out            = {full.Q_out/1e6:7.3f} MW")
print(f"util. efficiency = {full.eta_utilization*100:6.2f} %  (2nd-law, vs brine exergy)")
print(f"thermal eff.     = {full.eta_th*100:6.2f} %")
print()
print(f"energy residual  = {full.energy_balance_residual:.2e}")
print(f"exergy residual  = {full.exergy_balance_residual:.2e}")""")

md(r"""Exergy accounting also shows *where* the recoverable work is lost — which
components to improve first.""")

code(r"""ed = full.exergy_destruction
comp = list(ed.keys()); vals = [ed[c] / 1e3 for c in comp]
o = np.argsort(vals)[::-1]
comp = [comp[i] for i in o]; vals = [vals[i] for i in o]

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(comp, vals, color="#457b9d")
for i, v in enumerate(vals):
    ax.text(i, v, f"{v:.0f}", ha="center", va="bottom")
ax.set_ylabel("exergy destruction  [kW]")
ax.set_title("Where the available work is lost")
plt.tight_layout(); plt.show()""")

takeaway(r"""Energy and exergy balances close to machine precision, so the model
is internally honest — and the exergy ledger points at the evaporator and turbine
as the losses worth attacking.""")

# ============================================================================ #
# ACT 6 (balance of plant)
md(r"""## Act 6 · From cycle to grid — parasitics and the heat sink

The megawatts in Act 5 are what the *turbine* makes. The grid sees less, for two
reasons a cycle diagram never shows.

First, the plant spends power on itself. It must **reject** the condenser heat —
in an air-cooled binary unit that means banks of fans — and it must **move the
brine**, pumping it up the production well and back down the injection well.
These *parasitic loads* are a tax on gross output; air-cooled binary plants
typically deliver 75–90% of gross to the grid.

Second, the plant cannot choose its condensing temperature. That is set by the
**ambient air** the condenser rejects heat into, plus a fixed approach. So output
is not a single number — it **breathes with the weather**, strongest on a cold
night and weakest on a hot afternoon.""")

code(r"""from geothermal_orc import evaluate_plant, seasonal_performance

resource = GeothermalResource(T_reservoir_C=150.0, mass_flow=100.0)
# Design-ambient operating point (10 C -> ~30 C condensing, matching earlier acts).
pr = evaluate_plant("Isobutane", resource, ambient_C=10.0)

print(f"gross (turbine)       = {pr.W_gross/1e6:6.3f} MW")
print(f"  - working-fluid pump = {pr.W_pump_wf/1e6:6.3f} MW")
print(f"  - air-cooler fans    = {pr.P_fan/1e6:6.3f} MW")
print(f"  - brine pumping      = {pr.P_brine_pump/1e6:6.3f} MW")
print(f"  = net to grid        = {pr.W_net_plant/1e6:6.3f} MW   "
      f"({pr.net_gross_ratio*100:.0f}% of gross)")

# Waterfall from gross to net.
steps = [("gross", pr.W_gross, "#2a7d6f"),
         ("WF pump", -pr.W_pump_wf, "#bbbbbb"),
         ("air-cooler fans", -pr.P_fan, "#e76f51"),
         ("brine pumps", -pr.P_brine_pump, "#e9b44c")]
fig, ax = plt.subplots(figsize=(7.6, 4.2))
running = 0.0
for i, (lab, val, col) in enumerate(steps):
    bottom = running if val >= 0 else running + val
    ax.bar(i, abs(val) / 1e6, bottom=bottom / 1e6, color=col)
    running += val
ax.bar(len(steps), running / 1e6, color="#457b9d")
ax.set_xticks(range(len(steps) + 1))
ax.set_xticklabels([s[0] for s in steps] + ["net to grid"], rotation=20, ha="right")
ax.set_ylabel("power  [MW]")
ax.set_title("From turbine gross to net-to-grid (isobutane, 150 °C brine)")
plt.tight_layout(); plt.show()""")

md(r"""Now let the weather move. Feeding a representative monthly ambient profile
through the **same plant** (fixed evaporation temperature, condensing tracking
ambient) gives the seasonal output and the annual energy a developer sells.""")

code(r"""months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ambient_C = [2, 4, 8, 13, 18, 23, 26, 25, 20, 13, 6, 2]   # representative high-desert site

season = seasonal_performance("Isobutane", resource, ambient_C,
                              fixed_T_evap_C=pr.T_evap_C)
net_MW = season.W_net_plant / 1e6

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(months, net_MW, color="#457b9d")
ax.axhline(net_MW.mean(), color="#e76f51", ls="--",
           label=f"annual mean {net_MW.mean():.2f} MW")
ax.set_ylabel("net power to grid  [MW]")
ax.set_title("Seasonal output — same plant, condensing follows ambient")
ax.legend(); plt.tight_layout(); plt.show()

print(f"winter-to-summer swing : {net_MW.max():.2f} -> {net_MW.min():.2f} MW "
      f"({(1 - net_MW.min()/net_MW.max())*100:.0f}% drop)")
print(f"annual energy          : {season.annual_energy_MWh/1e3:.2f} GWh")
print(f"capacity factor (vs best month) : {season.capacity_factor*100:.0f} %")""")

takeaway(r"""Gross is not grid: parasitic fans and pumps take roughly a fifth off
the top, and because the condenser breathes ambient air the same hardware swings
substantially between winter and summer — the seasonality a fixed condensing
temperature hides.""")

# ============================================================================ #
# ACT 7
md(r"""## Act 7 · Tuning the design, and reconsidering the fluid

The 120 °C design point was a guess. Net power is **non-monotonic** in
evaporation temperature: raising it lifts cycle efficiency but tightens the
pinch, so the brine can boil less fluid. The best design sits between those
pulls. We sweep the evaporation temperature and overlay the optimizer.""")

code(r"""resource = GeothermalResource(T_reservoir_C=150.0, mass_flow=100.0)

T_sweep = np.linspace(45, 130, 16)
W_sweep = []
for Te in T_sweep:
    try:
        r = ORCCycle("Isobutane", T_evap_C=float(Te), T_cond_C=30.0
                     ).solve_with_resource(m_brine=100.0, T_brine_in_C=150.0,
                                           pinch_evap=5.0, n=40, n_search=24)
        W_sweep.append(r.W_net / 1e6)
    except Exception:
        W_sweep.append(np.nan)

opt = optimize_evaporation_temperature("Isobutane", resource, T_cond_C=30.0)
print(f"optimum: T_evap = {opt.T_evap_opt_C:.1f} °C, "
      f"W_net = {opt.W_net_opt/1e6:.3f} MW")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(T_sweep, W_sweep, "o-", color="#457b9d", label="swept")
ax.axvline(opt.T_evap_opt_C, color="#e76f51", ls="--",
           label=f"optimum {opt.T_evap_opt_C:.0f} °C")
ax.plot(opt.T_evap_opt_C, opt.W_net_opt / 1e6, "*", color="#e76f51",
        markersize=16)
ax.set_xlabel("evaporation temperature  [°C]")
ax.set_ylabel("net power  [MW]")
ax.set_title("Net power vs evaporation temperature (Isobutane, 150 °C brine)")
ax.legend(); plt.tight_layout(); plt.show()""")

md(r"""And the fluid choice deserves a second look now that we can score it on
*power for this resource* rather than on properties alone. Each candidate is
independently optimized over its own evaporation temperature, then ranked.""")

code(r"""candidates = ["Propane", "Isobutane", "n-Butane",
              "Isopentane", "n-Pentane", "R245fa"]
ranked = screen_fluids(candidates, resource, T_cond_C=30.0)

print(f"{'fluid':<12}{'T_evap [C]':>12}{'W_net [MW]':>12}{'feasible':>10}")
for r in ranked:
    print(f"{r.fluid:<12}{r.T_evap_opt_C:>12.1f}"
          f"{r.W_net_opt/1e6:>12.3f}{str(r.feasible):>10}")

feas = [r for r in ranked if r.feasible]
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh([r.fluid for r in feas][::-1],
        [r.W_net_opt / 1e6 for r in feas][::-1], color="#2a9d8f")
ax.set_xlabel("optimized net power  [MW]")
ax.set_title("Fluid screening on a 150 °C / 100 kg/s brine")
plt.tight_layout(); plt.show()""")

takeaway(r"""Tuning the boiling temperature and re-scoring fluids buys hundreds
of kilowatts — extra power from better design rather than a bigger resource. The
ranking is by raw power; safety and cost still get a vote.""")

# ============================================================================ #
# ACT 8
md(r"""## Act 8 · The limits reality imposes — silica and decline

Two constraints turn this point design into something that survives in the field.

**How cold may the brine be reinjected?** Cool brine holds less dissolved silica;
push it too cold and amorphous silica supersaturates and scales the injection
wells and piping shut. The floor comes from Fournier & Rowe (1977):

$$\log_{10} C = -\tfrac{731}{T} + 4.52 \quad\text{(amorphous, mg/kg, }T\text{ in K)}$$
$$\log_{10} C = -\tfrac{1309}{T} + 5.19 \quad\text{(quartz)}$$

The package's solubilities are validated against published values below.""")

code(r"""checks = [
    ("amorphous", amorphous_silica_solubility, 25.0, "115–120"),
    ("amorphous", amorphous_silica_solubility, 100.0, "~370"),
    ("quartz",    quartz_solubility,           100.0, "48–52"),
    ("quartz",    quartz_solubility,           150.0, "~125"),
]
print(f"{'curve':<11}{'T [C]':>7}{'computed':>11}{'literature':>14}")
for name, fn, Tc, lit in checks:
    val = fn(Tc + 273.15)
    print(f"{name:<11}{Tc:>7.0f}{val:>11.1f}{lit:>14}")

# Round-trip the quartz geothermometer: C(200C) -> T should recover 200C.
C200 = quartz_solubility(200 + 273.15)
print(f"\nquartz geothermometer round-trip: "
      f"C(200°C)={C200:.1f} mg/kg -> T={quartz_geothermometer(C200):.1f} °C")""")

code(r"""TC = np.linspace(20, 250, 200)
amorph = [amorphous_silica_solubility(T + 273.15) for T in TC]
quartz = [quartz_solubility(T + 273.15) for T in TC]

rich = GeothermalResource(T_reservoir_C=200.0, mass_flow=100.0)
C_res = rich.silica_mgkg
T_floor = rich.min_reinjection_temperature(1.0)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(TC, amorph, color="#e76f51", lw=2, label="amorphous silica")
ax.plot(TC, quartz, color="#264653", lw=2, label="quartz")
ax.axhline(C_res, color="#457b9d", ls="--",
           label=f"reservoir silica ≈ {C_res:.0f} mg/kg")
ax.axvline(T_floor, color="#2a9d8f", ls=":",
           label=f"reinjection floor ≈ {T_floor:.0f} °C")
ax.set_xlabel("temperature  [°C]"); ax.set_ylabel("SiO$_2$ solubility  [mg/kg]")
ax.set_title("Silica solubility and the amorphous-saturation reinjection limit")
ax.legend(); plt.tight_layout(); plt.show()

print(f"200 °C reservoir, quartz-equilibrium silica = {C_res:.1f} mg/kg")
print(f"amorphous-saturation (SI=1) reinjection floor = {T_floor:.1f} °C")""")

md(r"""**And the resource does not last forever.** As heat is mined the
production temperature drifts down. Snyder et al. (2017), analysing operating US
plants, found most wells decline roughly **linearly**, near **0.5 %/yr for
binary** and **0.8 %/yr for flash**.

How the *power* fades depends on what you assume about the plant. The optimistic
view **re-optimizes** a fresh plant at each year's temperature. The realistic
view runs the **installed year-0 hardware off-design** — a fixed turbine
(modelled with a Stodola swallowing law) and a fixed evaporator conductance,
with the turbine losing efficiency at part load. We show both, net-to-grid.""")

code(r"""from geothermal_orc import decline_curves

years = np.arange(0, 31)
binary = GeothermalResource(T_reservoir_C=150.0, mass_flow=100.0,
                            decline_rate=0.005, decline_mode="linear")
flash = GeothermalResource(T_reservoir_C=150.0, mass_flow=100.0,
                           decline_rate=0.008, decline_mode="linear")
T_bin = [binary.temperature_at(y) for y in years]
T_fl = [flash.temperature_at(y) for y in years]

# Net-to-grid over field life: re-optimized envelope vs fixed installed plant.
sample_years = [0, 10, 20, 30]
_design, T_samp, W_reopt, W_fixed = decline_curves(
    "Isobutane", resource, sample_years, ambient_C=10.0, decline_rate=0.005)
P_years = W_reopt / 1e6          # re-optimized net-to-grid, used in the verdict

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
a1.plot(years, T_bin, color="#2a9d8f", lw=2, label="binary 0.5 %/yr")
a1.plot(years, T_fl, color="#e76f51", lw=2, label="flash 0.8 %/yr")
a1.set_xlabel("year"); a1.set_ylabel("production temperature  [°C]")
a1.set_title("Resource thermal decline (Snyder et al., 2017)"); a1.legend()

a2.plot(sample_years, W_reopt / 1e6, "o-", color="#457b9d", lw=2,
        label="re-optimized each year")
a2.plot(sample_years, W_fixed / 1e6, "s--", color="#8b3a1f", lw=2,
        label="fixed plant, off-design")
a2.set_xlabel("year"); a2.set_ylabel("net power to grid  [MW]")
a2.set_title("Binary plant power as the resource cools"); a2.legend()
plt.tight_layout(); plt.show()

print("year :  T[C]  ->  re-opt[MW]  fixed[MW]")
for y, t, a, b in zip(sample_years, T_samp, W_reopt, W_fixed):
    print(f"{y:4d} : {t:6.1f}  -> {a/1e6:8.2f}  {b/1e6:8.2f}")""")

takeaway(r"""Silica sets *how cold* you may run the brine; decline sets *how
long* the power lasts. Both bound the design as firmly as the thermodynamics
do.""")

# ============================================================================ #
# ACT 9
md(r"""## Act 9 · The verdict for our 150 °C well

Pulling the threads together for the well we started with — 100 kg/s of 150 °C
brine, air-cooled at a 10 °C design ambient (~30 °C condensing), 5 K pinch. The
card states the headline **net-to-grid** design, what it costs in parasitics,
how much of the brine's available work it captures, where the brine leaves, and
how the output holds up over the field's life.""")

code(r"""win = ranked[0]
drop = (P_years[-1] / P_years[0] - 1) * 100.0

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.0),
                               gridspec_kw={"width_ratios": [1.25, 1]})
axL.axis("off"); axL.set_xlim(0, 1); axL.set_ylim(0, 1)
axL.text(0, 1.02, "Design verdict", fontsize=15, fontweight="bold",
         color="#8b3a1f", va="top")
rowiter = [
    ("Resource", "150 °C brine, 100 kg/s"),
    ("Working fluid", f"isobutane @ {pr.T_evap_C:.0f} °C evaporation"),
    ("Gross (turbine)", f"{pr.W_gross/1e6:.2f} MW"),
    ("Parasitics (fans + pumps)",
     f"{(pr.W_gross - pr.W_net_plant)/1e6:.2f} MW"),
    ("Net to grid",
     f"{pr.W_net_plant/1e6:.2f} MW  ({pr.net_gross_ratio*100:.0f}% of gross)"),
    ("Utilization efficiency",
     f"{pr.cycle.eta_utilization*100:.0f}% of brine exergy"),
    ("Brine reinjected at", f"{pr.cycle.brine_T_out-273.15:.0f} °C"),
    ("Net at year 30", f"{P_years[-1]:.2f} MW ({drop:+.0f}% vs year 0)"),
]
y = 0.90
for k, v in rowiter:
    axL.text(0.0, y, k, fontsize=10.5, color="#6b7280", va="top")
    axL.text(0.46, y, v, fontsize=11, color="#1a1f2b", va="top",
             fontweight="medium")
    y -= 0.125

axR.plot(sample_years, P_years, "o-", color="#457b9d", lw=2.2)
axR.fill_between(sample_years, P_years, color="#457b9d", alpha=0.08)
for x, p in zip(sample_years, P_years):
    axR.text(x, p + max(P_years) * 0.02, f"{p:.2f}", ha="center", fontsize=9,
             color="#1a1f2b")
axR.set_xlabel("year"); axR.set_ylabel("net to grid  [MW]")
axR.set_title("30-year outlook (re-optimized)", fontsize=11)
axR.set_ylim(0, max(P_years) * 1.2)
plt.tight_layout(); plt.show()""")

md(r"""So the honest answer to the opening question: this well drives a turbine
making roughly **4.5 MW gross**, but after the air-cooler fans and brine pumps,
about **3.5 MW reaches the grid** — near **77% of gross** — capturing close to
half of the brine's available work. It reinjects comfortably above the silica
floor, and over thirty years the net output falls by more than **40%** as the
reservoir cools (note the leverage: a ~15% drop in resource temperature drives a
far steeper drop in power, because available work scales faster than
temperature). **Propane** edges isobutane on raw cycle power — its lower critical
temperature lets the brine boil more fluid — but its A3 flammability is a real
plant-safety cost, exactly the kind of trade a screen surfaces and an engineer,
not an optimizer, decides.""")

# ============================================================================ #
# ACT 10 (Tier 2: cycle refinements)
md(r"""## Act 10 · A smarter cycle? Recuperation, pressure drops, and the cost of area

The verdict above used an idealized subcritical cycle. Before trusting it, an
engineer asks three questions a teaching model usually skips.

**Can we recover the turbine's leftover heat?** A dry fluid leaves the turbine
still superheated; a *recuperator* uses that to preheat the pump outlet. It
reliably lifts cycle efficiency — but watch what it does on a *free* heat
source.""")

code(r"""# Recuperator effectiveness sweep (dry fluid: n-pentane).
eps_grid = [0.0, 0.3, 0.6, 0.85]
eta_th, brine_out, wnet = [], [], []
for e in eps_grid:
    c = ORCCycle("n-Pentane", T_evap_C=120.0, T_cond_C=30.0,
                 recuperator_effectiveness=e)
    eta_th.append(c.solve().eta_th * 100)
    res = c.solve_with_resource(m_brine=100.0, T_brine_in_C=150.0, pinch_evap=5.0)
    brine_out.append(res.brine_T_out - 273.15)
    wnet.append(res.W_net / 1e6)

fig, ax1 = plt.subplots(figsize=(7.6, 4.2))
ax1.plot(eps_grid, eta_th, "o-", color="#2a7d6f")
ax1.set_xlabel("recuperator effectiveness")
ax1.set_ylabel("cycle efficiency  [%]", color="#2a7d6f")
ax2 = ax1.twinx()
ax2.plot(eps_grid, brine_out, "s--", color="#8b3a1f")
ax2.set_ylabel("brine reinjection T  [°C]", color="#8b3a1f")
ax1.set_title("Recuperator: efficiency rises (teal), but the brine leaves hotter (red)")
plt.tight_layout(); plt.show()

for e, et, b, w in zip(eps_grid, eta_th, brine_out, wnet):
    print(f"eps={e:.2f}:  eta_th={et:5.2f}%   net={w:.3f} MW   brine_out={b:.1f} C")""")

md(r"""Cycle efficiency climbs, yet net power barely moves and the brine is
**reinjected hotter** — the recuperator just shifts preheating from the brine to
the exhaust. On a free resource that is worse utilization for no power gain. (In
a waste-heat plant, where the source is finite or paid-for, the same device pays
handsomely — context decides.)

**Two more realities:** heat exchangers are not isobaric, and the 5 K pinch was
never free.""")

code(r"""from geothermal_orc import pinch_area_tradeoff

# Pressure drops cost net power (pump lifts higher, turbine exhausts higher).
base = ORCCycle("Isobutane", T_evap_C=120.0, T_cond_C=30.0).solve()
drop = ORCCycle("Isobutane", T_evap_C=120.0, T_cond_C=30.0,
                dp_evap_frac=0.05, dp_cond_frac=0.10).solve()
print(f"isobutane net work:  {base.w_net/1e3:.1f} kJ/kg (isobaric)  ->  "
      f"{drop.w_net/1e3:.1f} kJ/kg with 5%/10% drops "
      f"({(drop.w_net/base.w_net-1)*100:.0f}%)")

# The pinch is really a capital decision: tighter -> more power, but more area.
pinches = [2, 4, 6, 8, 10, 12]
pin, W, UA = pinch_area_tradeoff("Isobutane", resource, T_evap_C=120.0,
                                 pinches_C=pinches)
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.plot(UA / 1e6, W / 1e6, "o-", color="#457b9d")
for p, u, w in zip(pin, UA, W):
    ax.annotate(f"{p:.0f} K", (u / 1e6, w / 1e6),
                textcoords="offset points", xytext=(6, -10), fontsize=8)
ax.set_xlabel("evaporator conductance UA  [MW/K]")
ax.set_ylabel("net power  [MW]")
ax.set_title("The pinch is a capital decision: power vs heat-exchanger area")
plt.tight_layout(); plt.show()""")

takeaway(r"""A recuperator buys cycle efficiency but, on a free source, mostly
reinjects the brine hotter for no extra power. Real pressure drops shave a few
percent off net work. And the pinch is not a free input — it is an area-versus-
power trade with sharply diminishing returns past a few kelvin.""")

# ============================================================================ #
# ACT 11 (Tier 2: advanced fluids and cycles)
md(r"""## Act 11 · A different fluid, a different cycle — mixtures and transcritical

Two final levers change the *shape* of the heat-transfer match, not just its
numbers.

**Zeotropic mixtures** boil and condense over a temperature *glide* instead of a
flat plateau, so they can track the brine's straight-line cooling. The pay-off
is strongest for low-enthalpy resources where no single pure fluid fits well —
so here we drop to a **120 °C** well.""")

code(r"""from geothermal_orc import (MixtureCycle, screen_compositions,
                             optimize_evaporation_temperature)

res120 = GeothermalResource(T_reservoir_C=120.0, mass_flow=100.0)
fracs = [0.30, 0.45, 0.60, 0.75, 0.90]   # isobutane mole fraction
screen = screen_compositions("Isobutane", "Isopentane", fracs, res120,
                             T_evap_dew_C=84.0, T_cond_mean_C=30.0)
W_mix = [s.W_net / 1e6 if s else float("nan") for s in screen]
pure_ib = optimize_evaporation_temperature("Isobutane", res120,
                                           T_cond_C=30.0).W_net_opt / 1e6

fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.plot([f * 100 for f in fracs], W_mix, "o-", color="#2a9d8f",
        label="isobutane/isopentane mixture")
ax.axhline(pure_ib, color="#8b3a1f", ls="--",
           label=f"best pure isobutane  {pure_ib:.2f} MW")
ax.set_xlabel("isobutane mole fraction  [%]")
ax.set_ylabel("net power  [MW]")
ax.set_title("Zeotropic glide beats the best pure fluid — 120 °C resource")
ax.legend(); plt.tight_layout(); plt.show()
print(f"best mixture {max(w for w in W_mix if w == w):.3f} MW  vs  "
      f"pure isobutane {pure_ib:.3f} MW")""")

md(r"""At 120 °C a tuned mixture wins by a few percent. (Re-run it at 150 °C and
the advantage vanishes — there a pure light alkane already fits, which is why
isobutane won Act 7.)

**Transcritical** cycles go the other way: pump *above* the critical pressure so
there is no boiling plateau at all, letting the heating curve hug the brine. The
prize is the hot end of the resource that subcritical propane — capped near its
96.7 °C critical temperature — simply cannot reach.""")

code(r"""from geothermal_orc import TranscriticalCycle

P_grid = [48e5, 55e5, 65e5, 75e5, 90e5]
W_tc = []
for Ph in P_grid:
    r = TranscriticalCycle("Propane", P_high=Ph, T_turb_in_C=140.0,
                           T_cond_C=30.0).solve_with_resource(
        m_brine=100.0, T_brine_in_C=150.0, pinch_heater=5.0)
    W_tc.append(r.W_net / 1e6)
sub = optimize_evaporation_temperature("Propane", resource,
                                       T_cond_C=30.0).W_net_opt / 1e6

fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.plot([p / 1e5 for p in P_grid], W_tc, "o-", color="#457b9d",
        label="transcritical propane")
ax.axhline(sub, color="#8b3a1f", ls="--",
           label=f"best subcritical propane  {sub:.2f} MW")
ax.set_xlabel("turbine-inlet pressure  [bar]")
ax.set_ylabel("net power  [MW]")
ax.set_title("Transcritical propane unlocks the hot end — 150 °C resource")
ax.legend(); plt.tight_layout(); plt.show()
print(f"best transcritical {max(W_tc):.3f} MW  vs  best subcritical {sub:.3f} MW "
      f"(+{(max(W_tc)/sub-1)*100:.0f}%)")""")

takeaway(r"""Neither lever is a free win — both add pressure, surface, and
complexity. But each reshapes the temperature match: a zeotropic glide pays off
for low-enthalpy resources (~5% at 120 °C), and a transcritical cycle pays off at
150 °C (~5% with propane) by reaching a turbine inlet no subcritical fluid can.
Which one wins is set by the resource, not by a rule of thumb.""")

# ============================================================================ #
# ACT 12 (Tier 3: techno-economics)
md(r"""## Act 12 · The cost of a megawatt-hour

Every act so far has chased *power*. But a developer does not sell megawatts;
they sell megawatt-hours, and they live or die by the **cost** of those
megawatt-hours. So we turn the design into money: size every component, cost it
with the Turton module method (the standard in ORC techno-economics), add the
geothermal **wells** and operations, and annualize with a capital-recovery
factor to get the **levelized cost of electricity (LCOE)**.""")

code(r"""from geothermal_orc.economics import (levelized_cost, pinch_lcoe_tradeoff,
                                     lcoe_sensitivity, EconomicAssumptions)

econ = levelized_cost("Isobutane", resource, ambient_C=10.0, T_evap_C=95.0)

comp = econ.component_costs
indirects = econ.capex_surface - sum(comp.values())
labels = list(comp.keys()) + ["indirects", "wells"]
vals = [comp[k] / 1e6 for k in comp] + [indirects / 1e6, econ.capex_wells / 1e6]
colors = ["#2a7d6f", "#457b9d", "#bbbbbb", "#e76f51", "#cbb994", "#8b3a1f"]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(labels, vals, color=colors[:len(labels)])
ax.set_ylabel("CAPEX  [USD million]")
ax.set_title(f"Where the money goes — \${econ.capex_total/1e6:.0f}M total, "
             f"LCOE \${econ.lcoe:.0f}/MWh")
plt.tight_layout(); plt.show()

cap = (econ.crf * econ.capex_total) / econ.annual_energy_MWh
om = econ.annual_cost / econ.annual_energy_MWh - cap
print(f"net power      {econ.W_net_plant/1e6:.2f} MW")
print(f"specific CAPEX ${econ.specific_capex:,.0f}/kW   (wells "
      f"{econ.capex_wells/econ.capex_total*100:.0f}% of total)")
print(f"LCOE           {econ.lcoe:.1f} USD/MWh   (capital {cap:.0f}, O&M {om:.0f})")""")

md(r"""That is one design. The deeper question is whether the **cheapest** design
is the one that makes the **most power**. Recall the pinch from Act 10: a tighter
pinch buys power but costs heat-exchanger area. Whether that is worth it depends
on what your capital is tied up in — so we sweep the pinch for two resources: one
with expensive wells (deep, the usual geothermal case) and one with cheap wells
(shallow, or a retrofit where the surface plant dominates).""")

code(r"""pinches = [2, 3, 4, 5, 6, 8, 10, 12]
_, W, L_deep = pinch_lcoe_tradeoff("Isobutane", resource, 10.0, pinches,
                                   T_evap_C=95.0)
_, _, L_shallow = pinch_lcoe_tradeoff(
    "Isobutane", resource, 10.0, pinches,
    EconomicAssumptions(well_cost=1.5e6, n_wells=2), T_evap_C=95.0)

fig, ax1 = plt.subplots(figsize=(8, 4.4))
ax1.plot(pinches, L_deep, "o-", color="#8b3a1f",
         label="LCOE — expensive wells (\$5M x3)")
ax1.plot(pinches, L_shallow, "s-", color="#2a9d8f",
         label="LCOE — cheap wells (\$1.5M x2)")
for L, c in [(L_deep, "#8b3a1f"), (L_shallow, "#2a9d8f")]:
    i = int(np.nanargmin(L))
    ax1.scatter([pinches[i]], [L[i]], s=160, facecolors="none",
                edgecolors=c, linewidths=2.2, zorder=5)
ax1.set_xlabel("evaporator pinch  [K]"); ax1.set_ylabel("LCOE  [USD/MWh]")
ax2 = ax1.twinx()
ax2.plot(pinches, W / 1e6, ":", color="#457b9d", alpha=0.7)
ax2.set_ylabel("net power  [MW]  (dotted)", color="#457b9d")
ax1.set_title("Cost-optimal pinch (circled) vs the power-optimal tight pinch")
ax1.legend(loc="upper center"); plt.tight_layout(); plt.show()

print(f"power-optimal pinch : {pinches[int(np.argmax(W))]} K (tightest)")
print(f"cost-optimal pinch  : {pinches[int(np.nanargmin(L_deep))]} K "
      f"(expensive wells)  |  {pinches[int(np.nanargmin(L_shallow))]} K (cheap wells)")""")

md(r"""Finally, what actually moves the number? A tornado of the LCOE against the
assumptions that matter — the things a banker, not a thermodynamicist, worries
about.""")

code(r"""base = econ.lcoe
specs = [("well cost", "well_cost", (3e6, 7e6)),
         ("discount rate", "discount_rate", (0.06, 0.12)),
         ("capacity factor", "capacity_factor", (0.90, 0.60)),
         ("O&M (% CAPEX/yr)", "om_frac_capex", (0.02, 0.05))]
fig, ax = plt.subplots(figsize=(8, 3.6))
for i, (name, param, (lo, hi)) in enumerate(specs):
    v = lcoe_sensitivity("Isobutane", resource, 10.0, param, [lo, hi],
                         T_evap_C=95.0)
    ax.barh(i, v[1] - base, left=base, color="#e76f51", alpha=0.85)
    ax.barh(i, v[0] - base, left=base, color="#457b9d", alpha=0.85)
ax.axvline(base, color="#1a1f2b", lw=1.2)
ax.set_yticks(range(len(specs))); ax.set_yticklabels([s[0] for s in specs])
ax.set_xlabel("LCOE  [USD/MWh]")
ax.set_title(f"What moves the LCOE (base \${base:.0f}/MWh)")
plt.tight_layout(); plt.show()""")

takeaway(r"""Optimizing for power is not optimizing for cost. On a deep,
well-dominated resource the cheapest MWh comes from the **tightest pinch you can
afford** — squeeze the fixed wells for all the energy they can give. Only when
the surface plant dominates does a wider pinch win. And the LCOE is set first by
the wells, the capacity factor, and the cost of capital — which is exactly why a
marginal thermodynamic gain like recuperation, as the literature finds, rarely
pays for its extra hardware.""")

# ============================================================================ #
# ACT 13
md(r"""## Act 13 · Your turn — change the resource

The function below is the whole pipeline in miniature: give it a fluid, brine
temperature, condensing temperature, and pinch, and it finds a near-optimal
design and draws the evaporator. The static figure is the default case. **In
Colab (badge at the top), run all cells** and the next cell turns these inputs
into live sliders — GitHub's static viewer cannot run a kernel, so it shows a
fixed frame.""")

code(r"""def plot_design(fluid="Isobutane", T_brine=150.0, T_cond=30.0, pinch=5.0):
    # Quick near-optimal design for a 100 kg/s brine, with a T-Q plot.
    Tc_C = PropsSI("Tcrit", fluid) - 273.15
    lo, hi = T_cond + 5.0, min(Tc_C - 2.0, T_brine - pinch - 1.0)
    best, bestTe = None, None
    if hi > lo:
        for Te in np.linspace(lo, hi, 8):
            try:
                r = ORCCycle(fluid, T_evap_C=float(Te), T_cond_C=T_cond
                             ).solve_with_resource(
                                 m_brine=100.0, T_brine_in_C=T_brine,
                                 pinch_evap=pinch, n=40, n_search=24)
            except Exception:
                continue
            if best is None or r.W_net > best.W_net:
                best, bestTe = r, float(Te)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if best is None:
        ax.text(0.5, 0.5, "infeasible configuration", ha="center", va="center")
        ax.axis("off"); plt.show()
        print("No feasible design for these settings.")
        return
    ev = best.evaporator
    ax.plot(ev.duty / 1e6, ev.T_hot - 273.15, color="#e76f51", lw=2,
            label="brine (hot)")
    ax.plot(ev.duty / 1e6, ev.T_cold - 273.15, color="#2a9d8f", lw=2,
            label=f"{fluid} (cold)")
    ax.set_xlabel("cumulative duty  [MW]"); ax.set_ylabel("temperature  [°C]")
    ax.set_title(f"{fluid}: T_evap ≈ {bestTe:.0f} °C  →  "
                 f"net power {best.W_net/1e6:.2f} MW")
    ax.legend(); plt.tight_layout(); plt.show()
    print(f"net power         = {best.W_net/1e6:6.2f} MW")
    print(f"utilization eff.  = {best.eta_utilization*100:6.2f} %")
    print(f"brine outlet      = {best.brine_T_out-273.15:6.1f} °C")
    print(f"working-fluid flow= {best.m_wf:6.1f} kg/s   (T_evap ≈ {bestTe:.0f} °C)")


# Static default configuration (this frame is what GitHub shows).
plot_design()""")

# ============================================================================ #
md(r"""## How we know it is right — validation & references

The story is only as good as the numbers under it. This package is explicit
about what is *validated* versus what is *plausible*:

**Hard validations** (matched to published values):

* Amorphous and quartz silica solubilities reproduce Fournier & Rowe (1977) to
  within a few mg/kg, and the quartz geothermometer round-trips
  (200 °C → ~265 mg/kg → ~201 °C).
* Energy and exergy balances on every resource-coupled solve close to machine
  precision.
* Wet / dry / isentropic fluid classes agree with the literature consensus.
* **Benchmarked against the ORC literature** (see `tests/test_benchmarks.py`):
  subcritical thermal efficiency at 100 °C/30 °C lands at ~11–12% (vs the ~10%
  consensus of Saleh et al. 2007 and Su et al. 2018), ideal-cycle efficiency
  ~15%, and the 150 °C fluid screen tops out on isobutane/propane — matching
  Augustine et al. (NREL), who found isobutane optimal for 140–170 °C binary.
* **Advanced cycles preserve closure and match the literature direction:** the
  recuperated, pressure-drop, mixture, and transcritical solves all keep energy/
  exergy balances closed; zeotropic mixtures help low-enthalpy resources (~5% at
  120 °C) and transcritical propane helps at 150 °C (~5%), both consistent with
  the published consensus.
* **Techno-economics lands in range:** Turton module costing plus geothermal
  wells gives an LCOE of ~$90/MWh and ~$6,700/kW for the 150 °C base case —
  inside the published band for medium-temperature binary plants (Lazard, IRENA)
  — with wells the dominant (~60%+) share of CAPEX, as expected for geothermal.

**Plausibility checks** (right physics, right range — not matched to a named
plant):

* Absolute net power, utilization efficiency, and brine outlet temperatures fall
  within ranges DiPippo reports for comparable ~150 °C binary resources.
* Decline uses the Snyder et al. (2017) averages; an individual field can differ.

**References**

* R. DiPippo, *Geothermal Power Plants: Principles, Applications, Case Studies
  and Environmental Impact*, 4th ed., Butterworth-Heinemann, 2016.
* B. Saleh, G. Koglbauer, M. Wendland, J. Fischer, "Working fluids for
  low-temperature organic Rankine cycles," *Energy* 32 (2007) 1210–1221.
* R. O. Fournier, J. J. Rowe, "The solubility of amorphous silica in water at
  high temperatures and high pressures," *Am. Mineral.* 62 (1977) 1052–1056.
* D. M. Snyder, K. F. Beckers, K. R. Young, "Update on geothermal direct-use and
  power-plant thermal decline," *GRC Transactions* 41 (2017).
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python"},
}

with open("notebooks/demo.ipynb", "w") as fh:
    nbf.write(nb, fh)
print("wrote notebooks/demo.ipynb with", len(cells), "cells")
