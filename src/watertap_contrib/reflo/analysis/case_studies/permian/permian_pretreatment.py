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

reflo_dir = pathlib.Path(__file__).resolve().parents[3]
case_study_yaml = f"{reflo_dir}/data/technoeconomic/permian_case_study.yaml"
rho = 1125 * pyunits.kg / pyunits.m**3
rho_water = 995 * pyunits.kg / pyunits.m**3

solver = get_solver()

__all__ = [
    "build_permian_pretreatment",
    "set_operating_conditions",
    "add_treatment_costing",
    "set_permian_pretreatment_scaling",
    "init_system",
    "run_permian_pretreatment",
]


def build_permian_pretreatment(**kwargs):
    """
    Build Permian pretreatment flowsheet
    """

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.db = REFLODatabase()

    m.fs.properties = ZO(solute_list=["tds"])

    m.fs.properties_feed = SeawaterParameterBlock()
    m.fs.properties_vapor = SteamParameterBlock()

    # Begin building Treatment Block
    m.fs.treatment = treat = Block()

    treat.feed = Feed(property_package=m.fs.properties)
    treat.desal = StateJunction(property_package=m.fs.properties_feed)
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

    treat.disposal_ZO_mixer = Mixer(
        property_package=m.fs.properties,
        num_inlets=2,
        inlet_list=["ec_disposal", "cart_filt_disposal"],
        energy_mixing_type=MixingType.none,
        momentum_mixing_type=MomentumMixingType.none,
    )

    treat.disposal_SW_mixer = Mixer(
        property_package=m.fs.properties_feed,
        num_inlets=1,
        inlet_list=["zo_mixer"],
        # num_inlets=2,
        # inlet_list=["zo_mixer", "mvc_disposal"],
        energy_mixing_type=MixingType.none,
        momentum_mixing_type=MomentumMixingType.equality,
    )

    treat.chem_addition = FlowsheetBlock(dynamic=False)
    build_chem_addition(m, treat.chem_addition)

    treat.EC = FlowsheetBlock(dynamic=False)
    build_ec(m, treat.EC)

    treat.cart_filt = FlowsheetBlock(dynamic=False)
    build_cartridge_filtration(m, treat.cart_filt)

    treat.DWI = FlowsheetBlock(dynamic=False)
    build_dwi(m, treat.DWI, m.fs.properties_feed)

    # BUILD PRODUCT STREAM
    # feed > chem_addition > EC > cart_filt > ZO_to_SW_translator > desal unit > product
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

    treat.cart_filt_translated_to_desal = Arc(
        source=treat.zo_to_sw_feed.outlet, destination=treat.desal.inlet
    )

    treat.desal_to_product = Arc(
        source=treat.desal.outlet, destination=treat.product.inlet
    )

    # BUILD DISPOSAL STREAM
    #        EC > ZO_mixer > ZO_to_SW_translator > disposal_mixer > disposal_mixer > DWI
    # cart_filt > ZO_mixer
    #                                desal unit  > disposal_mixer

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

    treat.disposal_SW_mixer_to_dwi = Arc(
        source=treat.disposal_SW_mixer.outlet, destination=treat.DWI.feed.inlet
    )

    TransformationFactory("network.expand_arcs").apply_to(m)

    treat.recovery = Var(
        initialize=0.5,
        bounds=(0, 1.0001),
        units=pyunits.dimensionless,
        doc="Overall system recovery",
    )

    # m.fs.treatment.eq_recovery = Constraint(
    #     expr=m.fs.treatment.recovery * m.fs.treatment.feed.properties[0].flow_vol
    #     == m.fs.treatment.product.properties[0].flow_vol_phase["Liq"]
    # )

    add_treatment_costing(m)

    return m


def set_operating_conditions(m, Qin=5, tds=130, **kwargs):

    global flow_mass_water, flow_mass_tds, flow_in

    m.fs.properties.dens_mass_default = rho

    Qin = Qin * pyunits.Mgallons / pyunits.day
    flow_in = pyunits.convert(Qin, to_units=pyunits.m**3 / pyunits.s)
    flow_mass_water = pyunits.convert(Qin * rho_water, to_units=pyunits.kg / pyunits.s)
    flow_mass_tds = pyunits.convert(
        Qin * tds * pyunits.g / pyunits.liter, to_units=pyunits.kg / pyunits.s
    )

    m.fs.treatment.feed.properties[0].flow_mass_comp["H2O"].fix(flow_mass_water)
    m.fs.treatment.feed.properties[0].flow_mass_comp["tds"].fix(flow_mass_tds)
    m.fs.treatment.feed.properties[0].conc_mass_comp[...]

    set_chem_addition_op_conditions(m, m.fs.treatment.chem_addition, **kwargs)
    set_ec_operating_conditions(m, m.fs.treatment.EC, **kwargs)
    set_cart_filt_op_conditions(m, m.fs.treatment.cart_filt)


def add_treatment_costing(m):

    m.fs.treatment.costing = TreatmentCosting(case_study_definition=case_study_yaml)
    add_chem_addition_costing(
        m, m.fs.treatment.chem_addition, flowsheet_costing_block=m.fs.treatment.costing
    )
    add_ec_costing(m, m.fs.treatment.EC, flowsheet_costing_block=m.fs.treatment.costing)
    add_cartridge_filtration_costing(
        m, m.fs.treatment.cart_filt, flowsheet_costing_block=m.fs.treatment.costing
    )

    add_dwi_costing(
        m, m.fs.treatment.DWI, flowsheet_costing_block=m.fs.treatment.costing
    )

    m.fs.treatment.costing.cost_process()


def set_permian_pretreatment_scaling(m, calclate_m_scaling_factors=False, **kwargs):

    m.fs.properties.set_default_scaling(
        "flow_mass_comp",
        # 1 / value(flow_mass_water),
        1e-2,
        index=("H2O"),
    )

    m.fs.properties.set_default_scaling(
        "flow_mass_comp",
        # 1 / value(flow_mass_tds),
        0.1,
        index=("tds"),
    )

    m.fs.properties_feed.set_default_scaling(
        "flow_mass_phase_comp",
        0.1,
        index=("Liq", "TDS"),
    )

    m.fs.properties_feed.set_default_scaling(
        "flow_mass_phase_comp",
        1e-2,
        index=("Liq", "H2O"),
    )

    set_chem_addition_scaling(
        m, m.fs.treatment.chem_addition, calc_blk_scaling_factors=True
    )

    set_cart_filt_scaling(m, m.fs.treatment.cart_filt, calc_blk_scaling_factors=True)

    set_ec_scaling(m, m.fs.treatment.EC, calc_blk_scaling_factors=True)

    # set_mvc_scaling(m, m.fs.treatment.MVC, calc_blk_scaling_factors=True)

    set_scaling_factor(
        m.fs.treatment.product.properties[0].flow_mass_phase_comp["Liq", "H2O"], 1e2
    )
    set_scaling_factor(
        m.fs.treatment.product.properties[0].flow_mass_phase_comp["Liq", "TDS"], 1e5
    )

    # ZO to SW feed translator
    set_scaling_factor(
        m.fs.treatment.zo_to_sw_feed.properties_out[0].flow_mass_phase_comp[
            "Liq", "H2O"
        ],
        1e-2,
    )
    set_scaling_factor(
        m.fs.treatment.zo_to_sw_feed.properties_out[0].flow_mass_phase_comp[
            "Liq", "TDS"
        ],
        0.1,
    )

    # ZO to SW disposal translator
    set_scaling_factor(
        m.fs.treatment.zo_to_sw_disposal.properties_in[0].flow_mass_comp["H2O"],
        1,
    )
    set_scaling_factor(
        m.fs.treatment.zo_to_sw_disposal.properties_in[0].flow_mass_comp["tds"],
        1,
    )
    set_scaling_factor(
        m.fs.treatment.zo_to_sw_disposal.properties_out[0].flow_mass_phase_comp[
            "Liq", "H2O"
        ],
        1,
    )
    set_scaling_factor(
        m.fs.treatment.zo_to_sw_disposal.properties_out[0].flow_mass_phase_comp[
            "Liq", "TDS"
        ],
        1,
    )

    # ZO DISPOSAL MIXER
    # CF inlet
    set_scaling_factor(
        m.fs.treatment.disposal_ZO_mixer.cart_filt_disposal_state[0].flow_mass_comp[
            "H2O"
        ],
        100,
    )
    set_scaling_factor(
        m.fs.treatment.disposal_ZO_mixer.cart_filt_disposal_state[0].flow_mass_comp[
            "tds"
        ],
        1e8,
    )

    # EC inlet
    set_scaling_factor(
        m.fs.treatment.disposal_ZO_mixer.ec_disposal_state[0].flow_mass_comp["H2O"],
        1,
    )
    set_scaling_factor(
        m.fs.treatment.disposal_ZO_mixer.ec_disposal_state[0].flow_mass_comp["tds"],
        1,
    )

    # mixed state
    set_scaling_factor(
        m.fs.treatment.disposal_ZO_mixer.mixed_state[0].flow_mass_comp["H2O"],
        1,
    )
    set_scaling_factor(
        m.fs.treatment.disposal_ZO_mixer.mixed_state[0].flow_mass_comp["tds"],
        1,
    )
    # SW DISPOSAL MIXER
    # ZO mixer inlet
    set_scaling_factor(
        m.fs.treatment.disposal_SW_mixer.zo_mixer_state[0].flow_mass_phase_comp[
            "Liq", "H2O"
        ],
        1,
    )
    set_scaling_factor(
        m.fs.treatment.disposal_SW_mixer.zo_mixer_state[0].flow_mass_phase_comp[
            "Liq", "TDS"
        ],
        1,
    )

    # mixed state outlet
    set_scaling_factor(
        m.fs.treatment.disposal_SW_mixer.mixed_state[0].flow_mass_phase_comp[
            "Liq", "H2O"
        ],
        1e-2,
    )
    set_scaling_factor(
        m.fs.treatment.disposal_SW_mixer.mixed_state[0].flow_mass_phase_comp[
            "Liq", "TDS"
        ],
        0.1,
    )

    if calclate_m_scaling_factors:
        print("calclate_m_scaling_factors\n\n\n")
        calculate_scaling_factors(m)


def init_system(m, **kwargs):

    treat = m.fs.treatment

    treat.feed.initialize()
    propagate_state(treat.feed_to_chem_addition)

    init_chem_addition(m, treat.chem_addition)
    propagate_state(treat.chem_addition_to_ec)

    init_ec(m, treat.EC)
    propagate_state(treat.ec_to_cart_filt)
    propagate_state(treat.ec_to_disposal_mix)

    init_cart_filt(m, treat.cart_filt)
    propagate_state(treat.cart_filt_to_translator)
    propagate_state(treat.cart_filt_to_disposal_mix)

    treat.disposal_ZO_mixer.initialize()
    propagate_state(treat.disposal_ZO_mix_to_translator)

    treat.zo_to_sw_disposal.outlet.temperature[0].fix(320)
    treat.zo_to_sw_disposal.outlet.pressure[0].fix()
    treat.zo_to_sw_disposal.initialize()

    treat.zo_to_sw_feed.properties_out[0].temperature.fix(320)
    treat.zo_to_sw_feed.properties_out[0].pressure.fix()
    treat.zo_to_sw_feed.initialize()

    propagate_state(treat.cart_filt_translated_to_desal)

    treat.desal.properties[0].flow_vol
    treat.desal.initialize()

    propagate_state(treat.desal_to_product)

    propagate_state(treat.disposal_ZO_mix_translated_to_disposal_SW_mixer)
    # NOTE: variable that affects DOF in unclear way
    treat.disposal_SW_mixer.initialize()
    treat.disposal_SW_mixer.mixed_state[0].temperature.fix()
    # treat.disposal_SW_mixer.zo_mixer_state[0].pressure.fix()

    propagate_state(treat.disposal_SW_mixer_to_dwi)

    
    treat.DWI.unit.properties[0].flow_vol_phase
    treat.DWI.unit.properties[0].conc_mass_phase_comp

    # NOTE: variables that affect DOF in unclear way
    # treat.DWI.feed.properties[0].temperature.fix()
    # treat.DWI.feed.properties[0].pressure.fix()
    init_dwi(m, treat.DWI)

    
    treat.product.properties[0].flow_vol_phase
    treat.product.properties[0].conc_mass_phase_comp

    treat.product.initialize()


def run_permian_pretreatment():
    """
    Run Permian pretreatment flowsheet
    """

    m = build_permian_pretreatment()
    treat = m.fs.treatment

    set_operating_conditions(m)
    set_permian_pretreatment_scaling(m, calclate_m_scaling_factors=True)

    treat.feed.properties[0].flow_vol

    init_system(m)
    print(f"DOF = {degrees_of_freedom(m)}")

    treat.costing.initialize()
    flow_vol = treat.product.properties[0].flow_vol_phase["Liq"]
    treat.costing.electricity_cost.fix(0.07)
    treat.costing.add_LCOW(flow_vol)
    treat.costing.add_specific_energy_consumption(flow_vol, name="SEC")

    # NOTE: variables that affect DOF in unclear way
    # treat.chem_addition.unit.chemical_dosage.unfix()
    # treat.EC.unit.conductivity_constr.deactivate()
    # treat.EC.unit.conductivity.fix()

    print(f"DOF = {degrees_of_freedom(m)}")
    try:
        results = solver.solve(m)
    except ValueError:
        print_infeasible_constraints(m)
    assert_optimal_termination(results)

    # print(f"DOF TREATMENT BLOCK = {degrees_of_freedom(treat)}")
    # print(f"DOF FEED = {degrees_of_freedom(treat.feed)}")
    # print(f"DOF ZO TO SW FEED TB = {degrees_of_freedom(treat.zo_to_sw_feed)}")
    # print(f"DOF ZO TO SW DISPOSAL TB = {degrees_of_freedom(treat.zo_to_sw_disposal)}")
    # print(f"DOF SW TO ZO TB = {degrees_of_freedom(treat.sw_to_zo)}")

    # print(f"DOF CHEM ADDITION UNIT = {degrees_of_freedom(treat.chem_addition.unit)}")
    # print(f"DOF CHEM ADDITION FEED = {degrees_of_freedom(treat.chem_addition.feed)}")
    # print(f"DOF CHEM ADDITION PRODUCT = {degrees_of_freedom(treat.chem_addition.product)}")
    # print(f"DOF CHEM ADDITION DISPOSAL = {degrees_of_freedom(treat.chem_addition.disposal)}")

    # print(f"DOF EC UNIT = {degrees_of_freedom(treat.EC.unit)}")
    # print(f"DOF EC FEED = {degrees_of_freedom(treat.EC.feed)}")
    # print(f"DOF EC PRODUCT = {degrees_of_freedom(treat.EC.product)}")
    # print(f"DOF EC DISPOSAL = {degrees_of_freedom(treat.EC.disposal)}")

    # print(f"DOF CARTRIDGE FILTRATION UNIT = {degrees_of_freedom(treat.cart_filt.unit)}")
    # print(f"DOF CARTRIDGE FILTRATION FEED = {degrees_of_freedom(treat.cart_filt.feed)}")
    # print(f"DOF CARTRIDGE FILTRATION PRODUCT = {degrees_of_freedom(treat.cart_filt.product)}")
    # print(f"DOF CARTRIDGE FILTRATION DISPOSAL = {degrees_of_freedom(treat.cart_filt.disposal)}")

    # print(f"DOF DESAL STATE JUNCTION = {degrees_of_freedom(treat.desal)}")

    # print(f"DOF DISPOSAL ZO MIXER = {degrees_of_freedom(treat.disposal_ZO_mixer)}")

    # print(f"DOF DISPOSAL SW MIXER = {degrees_of_freedom(treat.disposal_SW_mixer)}")

    # print(f"DOF DWI UNIT = {degrees_of_freedom(treat.DWI.feed)}")
    # print(f"DOF DWI UNIT = {degrees_of_freedom(treat.DWI.unit)}")

    print(f"LCOW = {m.fs.treatment.costing.LCOW()}")

    return m


if __name__ == "__main__":

    m = run_permian_pretreatment()
    treat = m.fs.treatment

    print(f"DOF After Solve = {degrees_of_freedom(m)}")

    system_recovery = (
        treat.feed.properties[0].flow_vol() / treat.product.properties[0].flow_vol()
    )

    print(f"Pretreatment Recovery: {system_recovery:.2f}")

    print(
        f"Inlet flow_vol: {treat.feed.properties[0].flow_vol():.5f} {pyunits.get_units(treat.feed.properties[0].flow_vol)}"
    )
    print(
        f'Inlet TDS conc: {treat.feed.properties[0].conc_mass_comp["tds"]():.2f} {pyunits.get_units(treat.feed.properties[0].conc_mass_comp["tds"])}'
    )

    print(
        f'EC feed TDS conc: {treat.EC.feed.properties[0].conc_mass_comp["tds"]():.2f} {pyunits.get_units(treat.EC.feed.properties[0].conc_mass_comp["tds"])}'
    )

    print(
        f'EC product TDS conc: {treat.EC.product.properties[0].conc_mass_comp["tds"]():.2f} { pyunits.get_units(treat.EC.product.properties[0].conc_mass_comp["tds"])}'
    )

    print(
        f'EC disposal TDS conc: {treat.EC.disposal.properties[0].conc_mass_comp["tds"]():.2f} {pyunits.get_units(treat.EC.disposal.properties[0].conc_mass_comp["tds"])}'
    )

    print(
        f'EC feed flow_vol: {treat.EC.feed.properties[0].flow_vol():.2f} {pyunits.get_units(treat.EC.feed.properties[0].flow_vol)}'
    )

    print(
        f'EC product flow_vol: {treat.EC.product.properties[0].flow_vol():.2f} { pyunits.get_units(treat.EC.product.properties[0].flow_vol)}'
    )

    print(
        f'EC disposal flow_vol: {treat.EC.disposal.properties[0].flow_vol():.2f} {pyunits.get_units(treat.EC.disposal.properties[0].flow_vol)}'
    )


    print(
        f'CF product TDS conc: {treat.cart_filt.product.properties[0].conc_mass_comp["tds"]():.2f} {pyunits.get_units(treat.cart_filt.product.properties[0].conc_mass_comp["tds"])}'
    )

    print(
        f'CF product flow_vol: {treat.cart_filt.product.properties[0].flow_vol():.2f} {pyunits.get_units(treat.cart_filt.product.properties[0].flow_vol)}'
    )

    print(
        f'Product TDS conc: {treat.product.properties[0].conc_mass_phase_comp["Liq", "TDS"]():.2f} {pyunits.get_units(treat.product.properties[0].conc_mass_phase_comp["Liq", "TDS"]())}'
    )

    print(
        f'Product flow_vol: {treat.product.properties[0].flow_vol_phase["Liq"]():.2f} {pyunits.get_units(treat.product.properties[0].flow_vol_phase["Liq"])}'
    )

    print(
        f'Desal flow_vol: {treat.desal.properties[0].flow_vol_phase["Liq"]():.2f} {pyunits.get_units(treat.desal.properties[0].flow_vol_phase["Liq"])}'
    )

    print(
        f'DWI flow_vol: {treat.DWI.unit.properties[0].flow_vol_phase["Liq"]():.6f} {pyunits.get_units(treat.DWI.unit.properties[0].flow_vol_phase["Liq"])}'
    )

    print(
        f'DWI TDS conc: {treat.DWI.unit.properties[0].conc_mass_phase_comp["Liq", "TDS"]():.2f} {pyunits.get_units(treat.DWI.unit.properties[0].conc_mass_phase_comp["Liq", "TDS"])}'
    )
    print(f"DWI pressure: {treat.DWI.feed.properties[0].pressure()} Pa")

    print(
        f"Translator pressure: {treat.disposal_SW_mixer.zo_mixer_state[0].pressure()} Pa"
    )
    
    # treat.disposal_ZO_mixer.display()

    # treat.zo_to_sw_disposal.display()

    # treat.disposal_SW_mixer.display()

    # treat.DWI.unit.properties[0.0].display()