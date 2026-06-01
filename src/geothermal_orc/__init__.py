"""geothermal_orc: a literature-grade binary (Organic Rankine Cycle) geothermal
power-cycle simulator.

The package is organised in layers that mirror how a binary plant is analysed:

    thermo          - CoolProp-backed thermodynamic state objects and exergy.
    fluids          - curated working-fluid library with environmental criteria
                      and dry/wet/isentropic classification (Saleh et al., 2007).
    heat_exchanger  - LMTD, counter-flow pinch analysis and T-Q profiles.
    cycle           - subcritical binary ORC solver with energy/exergy balances
                      (DiPippo, 2016 conventions).
    geothermal      - resource model with silica (quartz/amorphous) solubility
                      and scaling limits (Fournier & Rowe, 1977) plus a thermal
                      decline model (Snyder et al., 2017).
    optimization    - evaporation-temperature optimisation and fluid screening.

References
----------
DiPippo, R. (2016). Geothermal Power Plants, 4th ed. Butterworth-Heinemann.
Saleh, B. et al. (2007). Working fluids for low-temperature organic Rankine
    cycles. Energy 32(7), 1210-1221.
Fournier, R.O. & Rowe, J.J. (1977). The solubility of amorphous silica in water
    at high temperatures and high pressures. Am. Mineral. 62, 1052-1056.
Snyder, D.M., Beckers, K.F., Young, K.R. & Johnston, B. (2017). Analysis of
    geothermal reservoir and well operational conditions ... GRC Transactions 41.
"""

from .thermo import State, specific_flow_exergy, DEAD_STATE_T, DEAD_STATE_P
from .fluids import Fluid, LIBRARY, get_fluid, screen, classify_slope
from .heat_exchanger import lmtd, HeatExchangerProfile, counterflow_profile
from .cycle import ORCCycle, CycleResult
from .geothermal import (
    amorphous_silica_solubility,
    quartz_solubility,
    quartz_geothermometer,
    silica_saturation_index,
    min_reinjection_temperature,
    GeothermalResource,
)
from .optimization import optimize_evaporation_temperature, screen_fluids
from .transcritical import TranscriticalCycle
from .economics import (
    EconomicAssumptions,
    EconomicResult,
    bare_module_cost,
    capital_recovery_factor,
    plant_capex,
    levelized_cost,
    pinch_lcoe_tradeoff,
    lcoe_sensitivity,
)
from .finance import (
    ProjectAssumptions,
    ProjectResult,
    project_cashflow,
    breakeven_ppa,
    monte_carlo,
)
from .mixtures import (
    MixtureCycle,
    MixtureCycleResult,
    mixture_string,
    bubble_dew,
    temperature_glide,
    screen_compositions,
)
from .plant import (
    PlantResult,
    SeasonalResult,
    evaluate_plant,
    seasonal_performance,
    air_cooled_fan_power,
    brine_pump_power,
    condensing_temperature,
    DesignPoint,
    design_plant,
    off_design_operation,
    decline_curves,
    profile_UA,
    part_load_turbine_efficiency,
    pinch_area_tradeoff,
)

__version__ = "0.1.0"

__all__ = [
    "State",
    "specific_flow_exergy",
    "DEAD_STATE_T",
    "DEAD_STATE_P",
    "Fluid",
    "LIBRARY",
    "get_fluid",
    "screen",
    "classify_slope",
    "lmtd",
    "HeatExchangerProfile",
    "counterflow_profile",
    "ORCCycle",
    "CycleResult",
    "amorphous_silica_solubility",
    "quartz_solubility",
    "quartz_geothermometer",
    "silica_saturation_index",
    "min_reinjection_temperature",
    "GeothermalResource",
    "optimize_evaporation_temperature",
    "screen_fluids",
    "PlantResult",
    "SeasonalResult",
    "evaluate_plant",
    "seasonal_performance",
    "air_cooled_fan_power",
    "brine_pump_power",
    "condensing_temperature",
    "DesignPoint",
    "design_plant",
    "off_design_operation",
    "decline_curves",
    "profile_UA",
    "part_load_turbine_efficiency",
    "pinch_area_tradeoff",
    "MixtureCycle",
    "MixtureCycleResult",
    "mixture_string",
    "bubble_dew",
    "temperature_glide",
    "screen_compositions",
    "TranscriticalCycle",
    "EconomicAssumptions",
    "EconomicResult",
    "bare_module_cost",
    "capital_recovery_factor",
    "plant_capex",
    "levelized_cost",
    "pinch_lcoe_tradeoff",
    "lcoe_sensitivity",
    "ProjectAssumptions",
    "ProjectResult",
    "project_cashflow",
    "breakeven_ppa",
    "monte_carlo",
    "__version__",
]
