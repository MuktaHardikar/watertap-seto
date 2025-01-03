import pathlib
from pyomo.environ import (
    ConcreteModel,
    value,
    TransformationFactory,
    Param,
    Var,
    Constraint,
    Set,
    Expression,
    Objective,
    NonNegativeReals,
    Block,
    RangeSet,
    check_optimal_termination,
    assert_optimal_termination,
    units as pyunits,
)
from pyomo.network import Arc, SequentialDecomposition
from pyomo.util.calc_var_value import calculate_variable_from_constraint as cvc

from idaes.core import FlowsheetBlock, UnitModelCostingBlock
import idaes.core.util.scaling as iscale
from idaes.core import MaterialFlowBasis
from idaes.core.util.scaling import (
    constraint_scaling_transform,
    calculate_scaling_factors,
    set_scaling_factor,
)
import idaes.logger as idaeslogger
from idaes.core.util.exceptions import InitializationError
from idaes.models.unit_models import (
    Product,
    Feed,
    StateJunction,
    Separator,
    Mixer,
    MixingType,
    MomentumMixingType,
)
from idaes.core.util.model_statistics import *
from idaes.core.util.initialization import propagate_state

from watertap.core.solvers import get_solver
from watertap_contrib.reflo.core.wt_reflo_database import REFLODatabase
from watertap.core.zero_order_properties import WaterParameterBlock as ZO


from watertap.core.util.model_diagnostics.infeasible import *
from watertap.core.util.initialization import *
from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock
from watertap.property_models.water_prop_pack import (
    WaterParameterBlock as SteamParameterBlock,
)
from watertap_contrib.reflo.costing import TreatmentCosting, EnergyCosting
from watertap_contrib.reflo.analysis.case_studies.permian import *
from watertap_contrib.reflo.analysis.case_studies.permian.components.MD import *

reflo_dir = pathlib.Path(__file__).resolve().parents[3]
case_study_yaml = f"{reflo_dir}/data/technoeconomic/permian_case_study.yaml"
rho = 1125 * pyunits.kg / pyunits.m**3
rho_water =  997 * pyunits.kg / pyunits.m**3

solver = get_solver()

__all__ = [
    "build_permian_pretreatment",
    "set_operating_conditions",
    "add_treatment_costing",
    "set_permian_scaling",
    "init_system",
    "run_permian_pretreatment",
]


def build_permian_st_md(Qin=5, Q_md=0.22478, Cin=118, water_recovery=0.2):
    """
    Build Permian pretreatment flowsheet
    """

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.properties_feed = SeawaterParameterBlock()

    # Begin building Treatment Block
    m.fs.treatment = treat = Block()

    treat.feed = Feed(property_package=m.fs.properties_feed)
    treat.disposal = Product(property_package=m.fs.properties_feed)
    treat.product = Product(property_package=m.fs.properties_feed)
    
    m.inlet_flow_rate = pyunits.convert(
        Q_md * pyunits.m**3 / pyunits.s, to_units=pyunits.m**3 / pyunits.s
    )
    m.inlet_salinity = pyunits.convert(
        Cin * pyunits.g / pyunits.liter, to_units=pyunits.kg / pyunits.m**3
    )

    m.water_recovery = water_recovery

    treat.md = FlowsheetBlock(dynamic=False)
    build_md(m, treat.md, m.fs.properties_feed)

    treat.DWI = FlowsheetBlock(dynamic=False)
    build_dwi(m, treat.DWI, m.fs.properties_feed)

    # BUILD PRODUCT STREAM
    # feed > chem_addition > EC > cart_filt > ZO_to_SW_translator > desal unit > product

    treat.feed_to_md = Arc(source=treat.feed.outlet, destination=treat.md.feed.inlet)

    treat.md_to_product = Arc(
        source=treat.md.permeate.outlet, destination=treat.product.inlet
    )

    # BUILD DISPOSAL STREAM
    #        EC > ZO_mixer > ZO_to_SW_translator > disposal_mixer > disposal_mixer > DWI
    # cart_filt > ZO_mixer
    #                                   MD unit  > disposal_mixer

    # treat.md_to_disposal = Arc(
    #     source=treat.md.concentrate.outlet, destination=treat.disposal.inlet
    # )

    treat.md_to_dwi = Arc(
        source=treat.md.concentrate.outlet, destination=treat.DWI.feed.inlet
    )

    TransformationFactory("network.expand_arcs").apply_to(m)


    add_treatment_costing(m)

    return m


def set_st_md_operating_conditions(m, Qin=5, tds=118, **kwargs):
    treat = m.fs.treatment

    feed_flow_rate = treat.md.model_input["feed_flow_rate"]
    feed_salinity = treat.md.model_input["feed_salinity"]
    feed_temp = treat.md.model_input["feed_temp"]

    treat.feed.properties.calculate_state(
        var_args={
            ("flow_vol_phase", "Liq"): pyunits.convert(
                feed_flow_rate * pyunits.L / pyunits.h,
                to_units=pyunits.m**3 / pyunits.s,
            ),
            ("conc_mass_phase_comp", ("Liq", "TDS")): feed_salinity,
            ("temperature", None): feed_temp + 273.15,
            ("pressure", None): 101325,
        },
        hold_state=True,
    )


def add_treatment_costing(m):
    m.fs.treatment.costing = TreatmentCosting(case_study_definition=case_study_yaml)

    add_dwi_costing(
        m, m.fs.treatment.DWI, flowsheet_costing_block=m.fs.treatment.costing
    )

    m.fs.treatment.md.unit.add_costing_module(m.fs.treatment.costing)
    m.fs.treatment.costing.cost_process()


def set_st_md_scaling(m, **kwargs):

    m.fs.properties.set_default_scaling(
        "flow_mass_comp",
        1 / value(flow_mass_water),
        index=("H2O"),
    )

    m.fs.properties.set_default_scaling(
        "flow_mass_comp",
        1 / value(flow_mass_tds),
        index=("tds"),
    )

    calculate_scaling_factors(m)


def init_st_md_system(m, **kwargs):

    treat = m.fs.treatment

    treat.feed.initialize()
    propagate_state(treat.feed_to_md)

    init_md(m, treat.md)
    
    propagate_state(treat.md_to_product) 
    treat.product.initialize()

    propagate_state(treat.md_to_dwi)
    init_dwi(m, treat.DWI)
    

def run_permian_st_md():
    """
    Run Permian pretreatment flowsheet
    """

    m = build_permian_st_md()
    treat = m.fs.treatment
    set_st_md_operating_conditions(m)

    init_st_md_system(m)

    print(f"DOF = {degrees_of_freedom(m)}")
    results = solver.solve(m)
    print_infeasible_constraints(m)
    assert_optimal_termination(results)

    print("\n--------- Before costing solve Completed ---------\n")

    add_treatment_costing(m)
    treat.costing.initialize()

    flow_vol = treat.product.properties[0].flow_vol_phase["Liq"]
    treat.costing.electricity_cost.fix(0.07)
    treat.costing.add_LCOW(flow_vol)
    treat.costing.add_specific_energy_consumption(flow_vol, name="SEC")

    results = solver.solve(m)
    print_infeasible_constraints(m)
    assert_optimal_termination(results)
    print("\n--------- After costing solve Completed ---------\n")
    

    return m


if __name__ == "__main__":

    m = run_permian_st_md()
    treat = m.fs.treatment
    report_MD(m, treat.md)
    print(f"DOF = {degrees_of_freedom(m)}")

    print(treat.product.properties[0].display())
    print(treat.DWI.unit.properties[0].display())

 


