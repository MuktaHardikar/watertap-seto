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
from pyomo.util.check_units import assert_units_consistent
from pyomo.util.calc_var_value import calculate_variable_from_constraint as cvc

from idaes.core import FlowsheetBlock, UnitModelCostingBlock
from idaes.core.util.initialization import propagate_state as _prop_state
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
from watertap.core import Database
from watertap_contrib.reflo.core.wt_reflo_database import REFLODatabase
from watertap.core.zero_order_properties import WaterParameterBlock as ZO
from watertap.property_models.multicomp_aq_sol_prop_pack import (
    MCASParameterBlock as MCAS,
)

# from watertap.costing.zero_order_costing import ZeroOrderCosting
from watertap.core.util.model_diagnostics.infeasible import *
from watertap.core.util.initialization import *
from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock
from watertap.property_models.water_prop_pack import (
    WaterParameterBlock as SteamParameterBlock,
)
from watertap_contrib.reflo.costing import (
    TreatmentCosting,
    EnergyCosting,
    REFLOCosting,
)
from watertap_contrib.reflo.analysis.case_studies.KBHDP.components.translator_4 import (
    Translator_ZO_TDS_to_TDS,
)
from watertap_contrib.reflo.analysis.case_studies.permian import *
from watertap_contrib.reflo.unit_models.deep_well_injection import DeepWellInjection

reflo_dir = pathlib.Path(__file__).resolve().parents[3]
case_study_yaml = f"{reflo_dir}/data/technoeconomic/permian_case_study.yaml"
rho = 1000 * pyunits.kg / pyunits.m**3


def build_permian():

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.db = REFLODatabase()

    m.fs.properties = ZO(solute_list=["tds"])

    m.fs.properties_feed = SeawaterParameterBlock()
    m.fs.properties_vapor = SteamParameterBlock()

    # Begin building Treatent Block
    m.fs.treatment = treat = Block()

    treat.feed = Feed(property_package=m.fs.properties)
    treat.product = Product(property_package=m.fs.properties_feed)

    # Add translator blocks
    treat.zo_to_sw_feed = Translator_ZO_to_SW(
        inlet_property_package=m.fs.properties,
        outlet_property_package=m.fs.properties_feed,
    )
    treat.zo_to_sw_disposal = Translator_ZO_to_SW(
        inlet_property_package=m.fs.properties,
        outlet_property_package=m.fs.properties_feed,
    )

    treat.sw_to_zo = Translator_SW_to_ZO(
        inlet_property_package=m.fs.properties_feed,
        outlet_property_package=m.fs.properties,
    )

    treat.disposal_ZO_mixer = Mixer(
        property_package=m.fs.properties,
        num_inlets=2,
        inlet_list=["ec_disposal", "cart_filt_disposal"],
        energy_mixing_type=MixingType.none,
        momentum_mixing_type=MomentumMixingType.none,
    )

    treat.disposal_SW_mixer = Mixer(
        property_package=m.fs.properties_feed,
        num_inlets=2,
        inlet_list=["zo_mixer", "mvc_disposal"],
        energy_mixing_type=MixingType.none,
        momentum_mixing_type=MomentumMixingType.none,
    )

    treat.chem_addition = FlowsheetBlock(dynamic=False)
    build_chem_addition(m, treat.chem_addition)

    treat.EC = FlowsheetBlock(dynamic=False)
    build_ec(m, treat.EC)

    treat.cart_filt = FlowsheetBlock(dynamic=False)
    build_cartridge_filtration(m, treat.cart_filt)

    treat.MVC = FlowsheetBlock(dynamic=False)
    build_mvc(m, treat.MVC)

    treat.DWI = FlowsheetBlock(dynamic=False)
    build_dwi(m, treat.DWI, m.fs.properties_feed)

    # BUILD PRODUCT STREAM
    # feed > chem_addition > EC > cart_filt > ZO_to_SW_translator > MVC > product
    treat.feed_to_chem_addition = Arc(
        source=treat.feed.outlet, destination=treat.chem_addition.feed.inlet
    )

    treat.chem_addition_to_ec = Arc(
        source=treat.chem_addition.product.outlet, destination=treat.EC.feed.inlet
    )

    treat.ec_to_cart_filt = Arc(
        source=treat.EC.product.outlet, destination=treat.cart_filt.feed.inlet
    )

    # from ZO to SW - feed
    treat.cart_filt_to_translator = Arc(
        source=treat.cart_filt.product.outlet, destination=treat.zo_to_sw_feed.inlet
    )

    treat.cart_filt_translated_to_mvc = Arc(
        source=treat.zo_to_sw_feed.outlet, destination=treat.MVC.feed.inlet
    )

    treat.mvc_to_product = Arc(
        source=treat.MVC.product.outlet, destination=treat.product.inlet
    )

    # BUILD DISPOSAL STREAM
    #        EC > ZO_mixer > ZO_to_SW_translator > disposal_mixer > disposal_mixer > DWI
    # cart_filt > ZO_mixer
    #                                        MVC > disposal_mixer

    treat.ec_to_disposal_mix = Arc(
        source=treat.EC.disposal.outlet, destination=treat.disposal_ZO_mixer.ec_disposal
    )

    treat.cart_filt_to_disposal_mix = Arc(
        source=treat.cart_filt.disposal.outlet,
        destination=treat.disposal_ZO_mixer.cart_filt_disposal,
    )

    treat.disposal_ZO_mix_to_translator = Arc(
        source=treat.disposal_ZO_mixer.outlet, destination=treat.zo_to_sw_disposal.inlet
    )

    treat.disposal_ZO_mix_translated_to_disposal_SW_mixer = Arc(
        source=treat.zo_to_sw_disposal.outlet,
        destination=treat.disposal_SW_mixer.zo_mixer,
    )

    treat.mvc_disposal_to_translator = Arc(
        source=treat.MVC.disposal.outlet,
        destination=treat.disposal_SW_mixer.mvc_disposal,
    )

    treat.disposal_SW_mixer_to_dwi = Arc(
        source=treat.disposal_SW_mixer.outlet, destination=treat.DWI.feed.inlet
    )

    TransformationFactory("network.expand_arcs").apply_to(m)

    add_treatment_costing(m)

    return m


def set_operating_conditions(m, Qin=5, tds=130):

    global flow_mass_water, flow_mass_tds

    Qin = Qin * pyunits.Mgallons / pyunits.day
    flow_in = pyunits.convert(Qin, to_units=pyunits.m**3 / pyunits.s)
    flow_mass_water = pyunits.convert(Qin * rho, to_units=pyunits.kg / pyunits.s)
    flow_mass_tds = pyunits.convert(
        Qin * tds * pyunits.g / pyunits.liter, to_units=pyunits.kg / pyunits.s
    )

    m.fs.treatment.feed.properties[0].flow_mass_comp["H2O"].fix(flow_mass_water)
    m.fs.treatment.feed.properties[0].flow_mass_comp["tds"].fix(flow_mass_tds)
    # m.fs.treatment.feed.properties[0].temperature.fix(273.15 + 50.52)  # K
    # m.fs.treatment.feed.properties[0].pressure.fix(1e5)  # Pa
    m.fs.treatment.feed.properties[0].conc_mass_comp[...]

    set_chem_addition_op_conditions(m, m.fs.treatment.chem_addition)
    set_ec_operating_conditions(m, m.fs.treatment.EC)
    set_cart_filt_op_conditions(m, m.fs.treatment.cart_filt)
    set_mvc_operating_conditions(m, m.fs.treatment.MVC)


def add_treatment_costing(m):
    m.fs.treatment.costing = TreatmentCosting(case_study_definition=case_study_yaml)
    add_chem_addition_costing(
        m, m.fs.treatment.chem_addition, flowsheet_costing_block=m.fs.treatment.costing
    )
    add_ec_costing(m, m.fs.treatment.EC, flowsheet_costing_block=m.fs.treatment.costing)
    add_cartridge_filtration_costing(
        m, m.fs.treatment.cart_filt, flowsheet_costing_block=m.fs.treatment.costing
    )
    add_mvc_costing(
        m, m.fs.treatment.MVC, flowsheet_costing_block=m.fs.treatment.costing
    )
    add_dwi_costing(
        m, m.fs.treatment.DWI, flowsheet_costing_block=m.fs.treatment.costing
    )


def set_permian_scaling(m, **kwargs):

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

    set_ec_scaling(m, m.fs.treatment.EC, calc_blk_scaling_factors=True)

    set_mvc_scaling(m, m.fs.treatment.MVC, calc_blk_scaling_factors=True)

    calculate_scaling_factors(m)


def init_system(m, **kwargs):

    treat = m.fs.treatment

    treat.feed.initialize()
    propagate_state(treat.feed_to_chem_addition)

    init_chem_addition(m, treat.chem_addition)
    propagate_state(treat.chem_addition_to_ec)
    # treat.EC.unit.properties.display()
    print(f"dof chem_addition = {degrees_of_freedom(treat.chem_addition.unit)}")

    init_ec(m, treat.EC)
    # treat.EC.unit.display()
    print(f"dof EC = {degrees_of_freedom(treat.EC.unit)}")
    # assert False
    propagate_state(treat.ec_to_cart_filt)
    propagate_state(treat.ec_to_disposal_mix)

    init_cart_filt(m, treat.cart_filt)
    propagate_state(treat.cart_filt_to_translator)
    propagate_state(treat.cart_filt_to_disposal_mix)

    treat.disposal_ZO_mixer.initialize()
    propagate_state(treat.disposal_ZO_mix_to_translator)

    treat.zo_to_sw_disposal.outlet.temperature[0].fix(293.15)
    treat.zo_to_sw_disposal.outlet.pressure[0].fix(101325)
    treat.zo_to_sw_disposal.initialize()

    treat.zo_to_sw_feed.properties_out[0].temperature.fix(273.15 + 50.52)  # K
    treat.zo_to_sw_feed.properties_out[0].pressure.fix(101325)
    treat.zo_to_sw_feed.initialize()

    propagate_state(treat.cart_filt_translated_to_mvc)

    init_mvc(m, treat.MVC)
    propagate_state(treat.mvc_to_product)
    propagate_state(treat.mvc_disposal_to_translator)

    # treat.MVC.feed.display()
    # assert False

    propagate_state(treat.disposal_ZO_mix_translated_to_disposal_SW_mixer)
    treat.disposal_SW_mixer.initialize()
    propagate_state(treat.disposal_SW_mixer_to_dwi)
    # treat.disposal_SW_mixer.mixed_state[0].display()
    # assert False

    init_dwi(m, treat.DWI)

    treat.product.initialize()

    # m.fs.treatment.costing.initialize()


if __name__ == "__main__":
    m = build_permian()
    set_operating_conditions(m)
    set_permian_scaling(m)
    print(f"dof = {degrees_of_freedom(m)}")

    init_system(m)
    # print(f"dof = {degrees_of_freedom(m)}")

    # m.fs.treatment.DWI.display()

    # for b in m.fs.component_objects(Block, descend_into=True):
        
    #     if "_expanded" in b.name:
    #         continue
    #     if "costing" in b.name:
    #         continue
    
    #     print(b.name, degrees_of_freedom(b))

