from pyomo.environ import (
    Var,
    ConcreteModel,
    Objective,
    Param,
    Expression,
    Constraint,
    Block,
    log10,
    TransformationFactory,
    assert_optimal_termination,
    value,
    units as pyunits,
    NonNegativeReals,
)

from idaes.core.util.model_statistics import *
from pyomo.network import Arc
from idaes.core import FlowsheetBlock
from idaes.core.solvers.get_solver import get_solver
from idaes.models.unit_models import Product, Feed
from idaes.core.util.model_statistics import *
from idaes.core.util.scaling import (
    set_scaling_factor,
    calculate_scaling_factors,
    constraint_scaling_transform,
)
from idaes.core import UnitModelCostingBlock
from idaes.core.util.initialization import propagate_state
from idaes.core.solvers.get_solver import get_solver
import idaes.logger as idaeslog


# from thermal_storage import ThermalEnergyStorage
from watertap_contrib.reflo.unit_models.zero_dimensional.thermal_storage import ThermalEnergyStorage

# from flat_plate_physical import FlatPlatePhysical
from watertap_contrib.reflo.solar_models.zero_order.flat_plate_physical import FlatPlatePhysical


from idaes.models.unit_models.heat_exchanger import(  
    HeatExchanger,
    delta_temperature_lmtd_callback,
    delta_temperature_amtd_callback,
    HeatExchangerFlowPattern,

)
# Using a slightly modified version tp use the same property package across all the other models
from watertap_contrib.reflo.unit_models.zero_dimensional.heat_exchanger_ntu import (
    HeatExchangerNTU, 
    HXNTUInitializer,
)
from watertap.core.util.model_diagnostics.infeasible import *

from idaes.models.unit_models import Heater

from watertap.property_models.water_prop_pack import WaterParameterBlock

from idaes.core.util.scaling import (
    calculate_scaling_factors,
    constraint_scaling_transform,
    unscaled_variables_generator,
    unscaled_constraints_generator,
    badly_scaled_var_generator,
    list_badly_scaled_variables,
    constraint_autoscale_large_jac
)

def build_thermal_flowsheet(
        GHI = 900, 
        elec_price = 0.07,
        md_set_point = 80 + 273.15,
        md_outlet_temp = 55,
        mass_fr_md = 0.01,
        mass_fr_fpc = 0.05,
        mass_fr_tes_solar_hx = 0.05,
        mass_fr_tes_process = 0.1,
              
):
    
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic = False)
    m.fs.properties = WaterParameterBlock()

    # Include FPC model
    m.fs.fpc =  FlatPlatePhysical(
        property_package = m.fs.properties
    )

    # Include TES model
    m.fs.tes = ThermalEnergyStorage(
        property_package = m.fs.properties
    )

    # HX between FPC and TES

    m.fs.hx_solar = HeatExchangerNTU(
                hot_side_name = 'fpc',
                cold_side_name = 'tes',
                fpc = {"property_package": m.fs.properties},
                tes = {"property_package": m.fs.properties}
        )

    # HX between the TES and process
    m.fs.hx_process = HeatExchanger(
                hot_side = {"property_package": m.fs.properties},
                cold_side = {"property_package": m.fs.properties},
                delta_temperature_callback = delta_temperature_lmtd_callback,
                flow_pattern=HeatExchangerFlowPattern.countercurrent,
        )

    # Grid heater to meet process inlet temperature set point
    m.fs.grid_heater = Heater(
        property_package = m.fs.properties)



    