from pyomo.environ import (
    ConcreteModel,
    Objective,
    Expression,
    value,
    Var,
    Param,
    Constraint,
    Set,
    Var,
    Block,
    log,
    log10,
    units as pyunits,
    NonNegativeReals,
)

from pyomo.network import Port
from idaes.core.solvers.get_solver import get_solver

from idaes.core.util.model_statistics import *
from idaes.core.util.scaling import *

from idaes.core import FlowsheetBlock, UnitModelCostingBlock

from watertap.core.zero_order_costing import ZeroOrderCosting
from watertap.core.util.model_diagnostics.infeasible import *
from watertap.costing import WaterTAPCosting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    number_variables,
    number_total_constraints,
    number_unused_variables,
)

from copy import deepcopy
from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock
from watertap.property_models.water_prop_pack import WaterParameterBlock
from watertap_contrib.seto.property_models.basic_water_properties import (
    BasicWaterParameterBlock,
)

from watertap_contrib.seto.costing import  SETOWaterTAPCosting 

from watertap_contrib.seto.costing import (
    TreatmentCosting,
    EnergyCosting,
    SETOSystemCosting,
)

from watertap_contrib.seto.unit_models.zero_order.chemical_softening_zo import ChemicalSofteningZO
from watertap_contrib.seto.solar_models.surrogate.trough.trough_surrogate_DG import TroughSurrogate
from watertap_contrib.seto.unit_models.surrogate import MEDTVCSurrogate

from idaes.models.unit_models import (
    Mixer,
    Separator,
    Product,
    Translator,
    MomentumMixingType,
)


# function to get order of magnitude
def get_order(x):
    exp = round(np.log10(x))-1
    return exp


def build_pretreatment():
    # Flowsheet set up
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    component_list = ["Ca_2+","Alkalinity_2-","TDS",]
    
    # Define property packages
    m.fs.soft_prop  =  BasicWaterParameterBlock(solute_list=component_list)
   
    # Define blocks
    m.fs.treatment = Block()
    
    m.fs.treatment.soft = soft = ChemicalSofteningZO(
    property_package=m.fs.soft_prop, silica_removal= False,
    softening_procedure_type= 'single_stage_lime'
    )

    return m


def build_treatment(m):

    number_of_effects = 10
    m.fs.med_tvc_liq_prop = SeawaterParameterBlock()
    m.fs.med_tvc_vap_prop =  WaterParameterBlock()
   
    # Define blocks
    m.fs.treatment.med_tvc = med_tvc= MEDTVCSurrogate(
                property_package_liquid=m.fs.med_tvc_liq_prop,
                property_package_vapor=m.fs.med_tvc_vap_prop,
                number_effects= number_of_effects,
    )

    return m


def build_energy(m):
    m.fs.energy = Block()
    m.fs.energy.trough = trough = TroughSurrogate(heat_load_range = [10,200])

    return m

def set_pretreatment_operating_conditions(m,tds,recovery):

    soft = m.fs.treatment.soft

    # Setting softening operating conditions
    prop_in = soft.properties_in[0]
    prop_out = soft.properties_out[0]

    # Fixed feed flow rate for analysis
    q_in = 10 * 3785 * pyunits.m**3 / pyunits.day  # gpd->m3/d

    # Arbitrary components for softening unit model
    ca_in = 0.7 * pyunits.kg / pyunits.m**3  
    alk_in = 1.95 * pyunits.kg / pyunits.m**3  
    tds_in = tds * pyunits.kg / pyunits.m**3 
    co2_in  = 0.10844915 * pyunits.kg / pyunits.m**3 
    ca_eff = ca_in*0.1*2.5  #->The effluent concentration in CaCO3 equivalent and 90% removal

    prop_in.conc_mass_comp["Ca_2+"].fix(ca_in)
    prop_in.conc_mass_comp["Alkalinity_2-"].fix(alk_in)
    prop_in.conc_mass_comp["TDS"].fix(tds_in)

    prop_in.flow_vol.fix(q_in)

    soft.no_of_mixer.fix(1)
    soft.no_of_floc.fix(2)

    soft.retention_time_mixer.fix(0.4)
    soft.retention_time_floc.fix(25)
    soft.retention_time_sed.fix(130)
    soft.retention_time_recarb.fix(20)
    soft.sedimentation_overflow.fix()

    soft.vel_gradient_mix.fix(300)
    soft.vel_gradient_floc.fix(50)

    # Assuming all the liquid is recovered
    soft.frac_vol_recovery.fix(1)
    soft.removal_efficiency.fix(0)
    soft.ca_eff_target.fix(ca_eff)

    soft.CO2_CaCO3.fix(co2_in)  

    # Single stage lime inputs
    soft.excess_CaO.fix(0)
    soft.CO2_second_basin.fix(0)
    soft.MgCl2_dosing.fix(0)
    soft.Na2CO3_dosing.fix(0)
    
    print(f"Softening DOF = {degrees_of_freedom(m)}")

    return m
        

def set_treatment_operating_conditions(m,tds,recovery):

    med_tvc = m.fs.treatment.med_tvc

    # Setting MED-TVC known operating conditions

    feed = med_tvc.feed_props[0]
    cool = med_tvc.cooling_out_props[0]
    
    steam = med_tvc.heating_steam_props[0]
    motive = med_tvc.motive_steam_props[0]

    steam_temp = 70
    motive_pressure = 35
    recovery_ratio = recovery
    feed_temp = 25

    q_in = m.fs.treatment.soft.properties_out[0].flow_vol
    # Feed flow to MED-TVC same as feed flow to chemical softening
    feed_flow = pyunits.convert(q_in*m.fs.treatment.soft.frac_vol_recovery(), to_units = pyunits.m**3/pyunits.s)

    # Set scaling factors for mass flow rates
    m.fs.med_tvc_liq_prop.set_default_scaling(
                    "flow_mass_phase_comp", 1e-3, index=("Liq", "H2O")
                    )
    m.fs.med_tvc_liq_prop.set_default_scaling(
                    "flow_vol_phase", 1e-5, index=("Liq", "H2O")
                    )
    m.fs.med_tvc_liq_prop.set_default_scaling(
                    "flow_mass_phase_comp", 1e3, index=("Liq", "TDS")
                    )
    m.fs.med_tvc_vap_prop.set_default_scaling(
                    "flow_mass_phase_comp", 1e-2, index=("Liq", "H2O")
                    )
    m.fs.med_tvc_vap_prop.set_default_scaling(
                    "flow_mass_phase_comp", 1e-3, index=("Vap", "H2O")
                    )
    
    # set_scaling_factor(feed.flow_mass_phase_comp['Liq', 'H2O'],sf)
    set_scaling_factor(feed.flow_mass_phase_comp['Liq', 'TDS'],1e-2)
    # set_scaling_factor(feed.flow_vol_phase['Liq'],1e1)
    # set_scaling_factor(feed.conc_mass_phase_comp['Liq','TDS'],1e2)
    # set_scaling_factor(feed.mass_frac_phase_comp['Liq','TDS'],1e2)
    set_scaling_factor(feed.temperature,1e-2)
    set_scaling_factor(feed.pressure,1e-5)
            

    med_tvc.feed_props.calculate_state(
                    var_args={
                            ("flow_vol_phase", "Liq"): feed_flow,
                            ("conc_mass_phase_comp", ("Liq", "TDS")): m.fs.treatment.soft.properties_out[0].conc_mass_comp["TDS"],
                            ("temperature", None): feed_temp + 273.15,
                            # feed flow is at atmospheric pressure
                            ("pressure", None): 101325,
                    },
                    outlvl = idaeslog.INFO_LOW,
                    hold_state=True
                    
            )

    steam.temperature.fix(steam_temp + 273.15)
    set_scaling_factor(steam.pressure_sat,1e-5)

    med_tvc.heating_steam_props.calculate_state(
                    var_args={
                            ("pressure_sat", None): value(steam.pressure),
                    },
                    hold_state=True,
                    )

    steam.flow_mass_phase_comp["Vap", "H2O"].unfix()
    steam.flow_mass_phase_comp["Liq", "H2O"].unfix()

    sf = get_scaling_factor(steam.flow_mass_phase_comp["Vap", "H2O"])
    set_scaling_factor(steam.flow_mass_phase_comp["Vap", "H2O"],sf)

    
    set_scaling_factor(motive.pressure,1e-6)
    set_scaling_factor(motive.pressure_sat,1e-6)

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

    sf = get_scaling_factor(motive.flow_mass_phase_comp["Vap", "H2O"])
    set_scaling_factor(motive.flow_mass_phase_comp["Vap", "H2O"],sf)

    med_tvc.recovery_vol_phase[0, "Liq"].fix(recovery_ratio)

    calculate_scaling_factors(med_tvc)

    print(f"MED TVC DOF = {degrees_of_freedom(m)}")

    return m

def add_pretreatment_costing(m):
    
    soft = m.fs.treatment.soft
    m.fs.treatment.costing = TreatmentCosting()
        
    soft.costing = UnitModelCostingBlock(flowsheet_costing_block = m.fs.treatment.costing)
    m.fs.treatment.costing.cost_process()
    m.fs.treatment.costing.add_LCOW(m.fs.treatment.soft.properties_out[0].flow_vol)


def add_treatment_costing(m):
    
    med_tvc = m.fs.treatment.med_tvc
    dist = med_tvc.distillate_props[0]

    m.fs.treatment.costing = TreatmentCosting()    
    med_tvc.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.treatment.costing)

    # Treatment costing components
    m.fs.treatment.costing.factor_maintenance_labor_chemical.fix(0)
    m.fs.treatment.costing.factor_total_investment.fix(1)
    m.fs.treatment.costing.heat_cost.set_value(0.025)
    m.fs.treatment.costing.cost_process()
    m.fs.treatment.costing.add_LCOW(dist.flow_vol_phase["Liq"])


def add_energy_costing(m, solar_capacity):
    trough = m.fs.energy.trough

    med_tvc = m.fs.treatment.med_tvc
    dist = med_tvc.distillate_props[0]

    m.fs.energy.costing = EnergyCosting()
    trough.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.energy.costing)

    # Trough costing components
    m.fs.energy.costing.factor_maintenance_labor_chemical.fix(0)
    m.fs.energy.costing.factor_total_investment.fix(1)
    m.fs.energy.costing.heat_cost.set_value(-0.025)
    m.fs.energy.costing.cost_process()
    m.fs.energy.costing.add_LCOW(dist.flow_vol_phase["Liq"])

    m.fs.energy.med_tvc_heat_demand_constr_lower = Constraint(
                expr = m.fs.treatment.costing.aggregate_flow_heat * solar_capacity
                == -1 * m.fs.energy.costing.aggregate_flow_heat
                )

    m.fs.sys_costing = SETOSystemCosting()
    m.fs.sys_costing.cost_process()
    m.fs.sys_costing.add_LCOW(dist.flow_vol_phase["Liq"])


def set_energy_operating_conditions(m,tds,recovery):
    # Trough operating conditions
    hours_storage = 7
    trough = m.fs.energy.trough
    
    trough.hours_storage.fix(hours_storage)
    exp = get_order(m.fs.treatment.med_tvc.thermal_power_requirement())
    sf = 10**-exp
    print(sf)
 
    set_scaling_factor(m.fs.energy.trough.heat, 1e-3)
    set_scaling_factor(m.fs.energy.trough.heat_load, 1e-3)
    # set_scaling_factor(m.fs.energy.trough.heat_annual_scaled, 1e-9)
    
    # set_scaling_factor(m.fs.energy.trough.electricity_annual_scaled, 1e-2)
    set_scaling_factor(m.fs.energy.trough.electricity,1e-2)
    # constraint_autoscale_large_jac(m)
    # calculate_scaling_factors(trough)

    return m


def main(tds_in,recovery,solar_capacity,trough_flag = 0):

    solver = get_solver()

    # Add treatment flowsheet
    m = build_pretreatment()    
    m = set_pretreatment_operating_conditions(m,tds_in,recovery)
    add_pretreatment_costing(m)

    m.fs.treatment.soft.initialize()
    results = solver.solve(m.fs.treatment)

    pretreatment_lcow =  m.fs.treatment.costing.LCOW()

    m = build_treatment(m)
    m = set_treatment_operating_conditions(m,tds_in,recovery)
    m.fs.treatment.med_tvc.initialize()
    add_treatment_costing(m)
    
    m.fs.objective_lcow = Objective(expr=m.fs.treatment.costing.LCOW)
    results = solver.solve(m.fs.treatment)

    treatment_lcow =  m.fs.treatment.costing.LCOW()
    
    if trough_flag:
        # Add energy flowsheet
        m = build_energy(m)
        m = set_energy_operating_conditions(m,tds_in,recovery)
        
        trough = m.fs.energy.trough 
        add_energy_costing(m,solar_capacity)
        calculate_scaling_factors(m.fs.energy)

        trough.initialize_build()
        m.fs.objective_lcow = Objective(expr=m.fs.sys_costing.LCOW)
        results = solver.solve(m)
    
    print(f"DOF = {degrees_of_freedom(m)}")

    print('Pretreatment LCOW:',pretreatment_lcow )
    print('Pre+Treatment LCOW:', treatment_lcow )
    print('Solar LCOW:', m.fs.energy.costing.LCOW())
    print('System LCOW:', m.fs.sys_costing.LCOW())
    print('Solar energy:', m.fs.energy.trough.heat_load())
    print('Specific heat consumption:', m.fs.treatment.med_tvc.specific_energy_consumption_thermal())

    return [m,pretreatment_lcow,treatment_lcow,m.fs.energy.costing.LCOW(),m.fs.sys_costing.LCOW()]

def create_results_table():

    cols = ['tds (kg/m3)', 'recovery', 'capex']
    df = pd.DataFrame(columns=cols)

    tds_sweep = np.array([30,40,50,60]) #g/L
    recovery = np.array([0.3,0.34,0.36,0.4])
    solar_capacity = np.array([0.5, 1, 1.5, 2])

    for tds in tds_sweep:
        for rec in recovery:
            print(tds,rec)
            [m,pretreatment_lcow,treatment_lcow,energy_lcow,sys_lcow] = main(tds,rec,2,1)

            temp = {'tds (kg/m3)': tds,
                    'recovery' : rec, 
                    'pretreatment lcow ($/m3)': pretreatment_lcow,
                    'treatment lcow ($/m3)': treatment_lcow,
                    'energy lcow ($/m3)': energy_lcow,
                    'system lcow ($/m3)': sys_lcow,
                    'capex':m.fs.sys_costing.total_capital_cost()
                    }

            df = df.append(temp, ignore_index=True)

    return df


if __name__ == "__main__": 
    tds_in = 40 # g/L
    recovery = 0.4
    solar_capacity = 1  # Fraction represents capacity of trough in comparison to process heat demand
    trough_flag = 1 # Include trough : 1, Exclude trough: 0

    [m,pretreatment_lcow,treatment_lcow,energy_lcow,sys_lcow] = main(tds_in,recovery,solar_capacity,trough_flag)


# df = create_results_table()