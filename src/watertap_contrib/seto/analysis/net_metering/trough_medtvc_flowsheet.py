from pyomo.environ import (
    ConcreteModel,
    value,
    assert_optimal_termination,
    units as pyunits,
    Block,
    Constraint
)
import re
from pyomo.network import Port
from idaes.core import FlowsheetBlock, UnitModelCostingBlock
from watertap_contrib.seto.unit_models.surrogate import MEDTVCSurrogate

from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock
from watertap.property_models.water_prop_pack import WaterParameterBlock
from watertap_contrib.seto.costing import SETOWaterTAPCosting
from idaes.core.util.testing import initialization_tester
from watertap.core.util.initialization import assert_no_degrees_of_freedom
from pyomo.util.check_units import assert_units_consistent

from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import *
from idaes.core.util.testing import initialization_tester
from idaes.core.util.scaling import (
    calculate_scaling_factors,
    constraint_scaling_transform,
    unscaled_variables_generator,
    unscaled_constraints_generator,
    badly_scaled_var_generator,
)
from watertap.core.util.model_diagnostics.infeasible import *
from idaes.core.util.scaling import *

import idaes.logger as idaeslog
from watertap_contrib.seto.solar_models.surrogate.trough import TroughSurrogate
from watertap_contrib.seto.costing import (
    TreatmentCosting,
    EnergyCosting,
    SETOSystemCosting,
)


def build_trough_medtvc(
        number_of_effects = 12,
        feed_salinity = 35,
        feed_temperature = 25,
        steam_temp = 80,
        motive_pressure = 24,
        sys_capacity = 2000,
        recovery_ratio = 0.3,
        hours_storage = 10,
        add_trough = True,
):
        
        # Create concrete med-tvc model
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.liquid_prop = SeawaterParameterBlock()
        m.fs.vapor_prop = WaterParameterBlock()
        
        m.fs.treatment = Block()
        m.fs.treatment.med_tvc = MEDTVCSurrogate(
                property_package_liquid=m.fs.liquid_prop,
                property_package_vapor=m.fs.vapor_prop,
                number_effects= number_of_effects,  # assuming 12 effects by default
                )


        # Create state block variables
        med_tvc = m.fs.treatment.med_tvc
        feed = med_tvc.feed_props[0]
        cool = med_tvc.cooling_out_props[0]
        dist = med_tvc.distillate_props[0]
        steam = med_tvc.heating_steam_props[0]
        motive = med_tvc.motive_steam_props[0]

        feed_salinity = feed_salinity * pyunits.kg / pyunits.m**3
        feed_temperature = feed_temperature
        motive_pressure = motive_pressure
        sys_capacity = sys_capacity * pyunits.m**3 / pyunits.day
        recovery_ratio = recovery_ratio * pyunits.dimensionless
        feed_flow = pyunits.convert(
                        (sys_capacity / recovery_ratio), to_units=pyunits.m**3 / pyunits.s
                )  # feed volumetric flow rate [m3/s]
        

        # Set scaling factors for mass flow rates
        m.fs.liquid_prop.set_default_scaling(
                "flow_mass_phase_comp", 1e-2, index=("Liq", "H2O")
                )
        m.fs.liquid_prop.set_default_scaling(
                "flow_vol_phase", 1, index=("Liq", "H2O")
                )
        m.fs.liquid_prop.set_default_scaling(
                "flow_mass_phase_comp", 1e3, index=("Liq", "TDS")
                )
        m.fs.vapor_prop.set_default_scaling(
                "flow_mass_phase_comp", 1e-2, index=("Liq", "H2O")
                )
        m.fs.vapor_prop.set_default_scaling(
                "flow_mass_phase_comp", 1, index=("Vap", "H2O")
                )
        
        # Set scaling factors and calculate state for feed
        
        set_scaling_factor(feed.flow_vol_phase['Liq'],1e-4)
        set_scaling_factor(feed.conc_mass_phase_comp['Liq','TDS'],1e2)
        set_scaling_factor(feed.temperature,1e-1)
        set_scaling_factor(feed.pressure,1e-5)

        med_tvc.feed_props.calculate_state(
                var_args={
                        ("flow_vol_phase", "Liq"): feed_flow,
                        ("conc_mass_phase_comp", ("Liq", "TDS")): feed_salinity,
                        ("temperature", None): feed_temperature + 273.15,
                        # feed flow is at atmospheric pressure
                        ("pressure", None): 101325,
                },
                outlvl = idaeslog.INFO_LOW,
                hold_state=True
                
        )

        # Set scaling factors and calculate state for steam
        
        steam.temperature.fix(steam_temp + 273.15)
        set_scaling_factor(steam.pressure_sat,1e-5)

        med_tvc.heating_steam_props.calculate_state(
                var_args={
                        ("pressure_sat", None): value(steam.pressure),
                },
                hold_state=True,
                )

        # Release mass flow rate
        steam.flow_mass_phase_comp["Vap", "H2O"].unfix()
        steam.flow_mass_phase_comp["Liq", "H2O"].unfix()


        # Set scaling factors and calculate state for motive

        set_scaling_factor(motive.pressure,1e-5)
        set_scaling_factor(motive.pressure_sat,1e-5)

        med_tvc.motive_steam_props.calculate_state(
                var_args={
                        ("pressure", None): motive_pressure * 1e5,
                        ("pressure_sat", None): motive_pressure * 1e5,
                },
                hold_state=True,
                )
        
        # Release mass flow rate
        motive.flow_mass_phase_comp["Vap", "H2O"].unfix()
        motive.flow_mass_phase_comp["Liq", "H2O"].unfix()

        med_tvc.recovery_vol_phase[0, "Liq"].fix(recovery_ratio)

        calculate_scaling_factors(m)

        # Add treatment costing block

        m.fs.treatment.costing = TreatmentCosting()
        m.fs.treatment.med_tvc.costing = UnitModelCostingBlock(
                flowsheet_costing_block=m.fs.treatment.costing
        )
        m.fs.treatment.costing.factor_maintenance_labor_chemical.fix(0)
        m.fs.treatment.costing.factor_total_investment.fix(1)
        m.fs.treatment.costing.cost_process()
        m.fs.treatment.costing.add_LCOW(dist.flow_vol_phase["Liq"])

        med_tvc.initialize()
        
        # Add trough energy block

        m.fs.energy = Block()
        m.fs.energy.trough = trough = TroughSurrogate()
        trough.hours_storage.fix(hours_storage)

        # Add trough costing
        m.fs.energy.costing = EnergyCosting()
        m.fs.energy.trough.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.energy.costing
        )
        m.fs.energy.costing.factor_maintenance_labor_chemical.fix(0)
        m.fs.energy.costing.factor_total_investment.fix(1)
        m.fs.energy.costing.cost_process()
        m.fs.energy.costing.add_LCOW(dist.flow_vol_phase["Liq"])

        # Linking constraint
        # m.fs.energy.med_tvc_heat_demand_constr = Constraint(
        #         expr=trough.heat_load
        #         >= pyunits.convert(med_tvc.thermal_power_requirement, to_units=pyunits.MW)
        #         )

        m.fs.energy.lt_med_heat_demand_constr = Constraint(
            expr=m.fs.treatment.costing.aggregate_flow_heat
            <= -1 * m.fs.energy.costing.aggregate_flow_heat
        )

        set_scaling_factor(m.fs.energy.trough.heat_load, 1e-3)
        set_scaling_factor(m.fs.energy.trough.heat_annual, 1e-6)
        set_scaling_factor(m.fs.energy.trough.heat, 1e-1)

        set_scaling_factor(m.fs.energy.trough.electricity_annual, 1e-6)
        set_scaling_factor(m.fs.energy.trough.electricity, 1e-1)

        # Add system costing 
        m.fs.sys_costing = SETOSystemCosting()
        m.fs.sys_costing.add_LCOW(dist.flow_vol_phase["Liq"])

        calculate_scaling_factors(m)

        return m