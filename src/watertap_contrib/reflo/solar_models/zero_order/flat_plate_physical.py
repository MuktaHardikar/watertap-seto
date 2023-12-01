###############################################################################
# WaterTAP Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National
# Laboratory, National Renewable Energy Laboratory, and National Energy
# Technology Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#
###############################################################################

from copy import deepcopy
from pyomo.environ import Var, Param, Suffix, cos, sin, log, exp, units as pyunits
from pyomo.common.config import ConfigBlock, ConfigValue, In
from idaes.core import declare_process_block_class
# Import IDAES cores
from idaes.core import (
    ControlVolume0DBlock,
    declare_process_block_class,
    MaterialBalanceType,
    EnergyBalanceType,
    MomentumBalanceType,
    UnitModelBlockData,
    useDefault,
)
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.solvers import get_solver
import idaes.logger as idaeslog

from watertap_contrib.reflo.core import SolarEnergyBaseData

__author__ = "Matthew Boyd"
# Updated by Mukta Hardikar

@declare_process_block_class("FlatPlatePhysical")
class FlatPlatePhysicalData(SolarEnergyBaseData):
    """
    Physical model for flat plate
    based on equations in Solar Engineering of Thermal Processes, Duffie and Beckman, 4th ed.
    """

    CONFIG = SolarEnergyBaseData.CONFIG()


    CONFIG.declare(
        "property_package",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package to use for control volume",
            doc="""Property parameter object used to define property calculations,
    **default** - useDefault.
    **Valid values:** {
    **useDefault** - use default package from parent model or flowsheet,
    **PhysicalParameterObject** - a PhysicalParameterBlock object.}""",
        ),
    )

    CONFIG.declare(
        "property_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing property packages",
            doc="""A ConfigBlock with arguments to be passed to a property block(s)
    and used when constructing these,
    **default** - None.
    **Valid values:** {
    see property package for documentation.}""",
        ),
    )

    CONFIG.declare(
        "energy_balance_type",
        ConfigValue(
            default=EnergyBalanceType.useDefault,
            domain=In(EnergyBalanceType),
            description="Energy balance construction flag",
            doc="""Indicates what type of energy balance should be constructed,
            **default** - EnergyBalanceType.useDefault.
            **Valid values:** {
            **EnergyBalanceType.useDefault - refer to property package for default
            balance type
            **EnergyBalanceType.none** - exclude energy balances,
            **EnergyBalanceType.enthalpyTotal** - single enthalpy balance for material,
            **EnergyBalanceType.enthalpyPhase** - enthalpy balances for each phase,
            **EnergyBalanceType.energyTotal** - single energy balance for material,
            **EnergyBalanceType.energyPhase** - energy balances for each phase.}""",
        ),
    )

    CONFIG.declare(
        "momentum_balance_type",
        ConfigValue(
            default=MomentumBalanceType.pressureTotal,
            domain=In(MomentumBalanceType),
            description="Momentum balance construction flag",
            doc="""Indicates what type of momentum balance should be constructed,
            **default** - MomentumBalanceType.pressureTotal.
            **Valid values:** {
            **MomentumBalanceType.none** - exclude momentum balances,
            **MomentumBalanceType.pressureTotal** - single pressure balance for material,
            **MomentumBalanceType.pressurePhase** - pressure balances for each phase,
            **MomentumBalanceType.momentumTotal** - single momentum balance for material,
            **MomentumBalanceType.momentumPhase** - momentum balances for each phase.}""",
        ),
    )


    def build(self):
        super().build()

        # Add inlet state block
        tmp_dict = dict(**self.config.property_package_args)
        tmp_dict["has_phase_equilibrium"] = False
        tmp_dict["parameters"] = self.config.property_package
        tmp_dict["defined_state"] = True
        self.inlet_block = self.config.property_package.state_block_class(
            self.flowsheet().config.time,
            doc="Material properties of inlet stream",
            **tmp_dict,
        )

        # Add outlet state block
        tmp_dict = dict(**self.config.property_package_args)
        tmp_dict["has_phase_equilibrium"] = False
        tmp_dict["parameters"] = self.config.property_package
        tmp_dict["defined_state"] = False  # block is not an inlet
        self.outlet_block = self.config.property_package.state_block_class(
            self.flowsheet().config.time,
            doc="Material properties of outlet stream",
            **tmp_dict,
        )

        # Add Ports
        self.add_inlet_port(name='inlet',block = self.inlet_block)
        self.add_outlet_port(name='outlet',block = self.outlet_block)


        self.scaling_factor = Suffix(direction=Suffix.EXPORT)
        self._tech_type = "flat_plate"

        # ==========PARAMETERS==========
        self.area_coll = Param(
            initialize=1, units=pyunits.m**2, doc="area of a single collector"
        )

        self.FRta = Param(
            initialize=1,
            units=pyunits.dimensionless,
            doc="collector heat removal factor * effective transmittance-absorption product",
        )

        self.FRUL = Param(
            initialize=1,
            units=pyunits.W / (pyunits.m**2 * pyunits.K),
            doc="collector heat removal factor * collector heat loss coefficient",
        )

        # Not used
        # self.iam = Param(
        #     initialize=1,
        #     units=pyunits.dimensionless,
        #     doc="incidence angle modifier coefficient",
        # )

        self.mdot_test = Param(
            initialize=1,
            units=pyunits.kg / pyunits.s,
            doc="mass flow rate of fluid during characterization test",
        )

        self.cp_test = Param(
            initialize= 4184,
            units=pyunits.J / (pyunits.kg * pyunits.K),
            doc="specific heat capacity of fluid during characterization test",
        )

        self.cp_use = Param(
            initialize=4184,
            units=pyunits.J / (pyunits.kg * pyunits.K),
            doc="specific heat capacity of fluid during use",
        )

        self.ncoll = Param(
            initialize=1,
            units=pyunits.dimensionless,
            doc="number of collectors in array",
        )

        self.pump_watts = Param(initialize=1, units=pyunits.W, doc="pump power")

        self.pump_eff = Param(
            initialize=1, units=pyunits.dimensionless, doc="pump efficiency"
        )

        self.T_amb = Param(initialize=30+273.15, units=pyunits.K, doc="ambient temperature")

        self.G_trans = Param(
            initialize=900,
            units=pyunits.W / pyunits.m**2,
            doc="irradiance transmitted through glazing",
        )

        # ==========VARIABLES==========

        # self.mdot = Var(
        #     initialize = 1,
        #     units=pyunits.kg / pyunits.s,
        #     doc="mass flow rate of medium through FPC",
        # )

        # self.T_in = Var(self.flowsheet().config.time,initialize = 80+273.15, units=pyunits.K, doc="inlet temperature")

        # self.T_out = Var(self.flowsheet().config.time,initialize = 85+273.15,units=pyunits.K, doc="outlet temperature" )
        
        self.FprimeUL = Var(
            initialize=1,
            units=pyunits.W / (pyunits.m**2 * pyunits.K),
            doc="corrected collector heat loss coefficient, D&B Eq. 6.20.4",
        )

        self.r = Var(
            initialize=1,
            units=pyunits.dimensionless,
            doc="ratio of FRta_use to FRta_test, D&B Eq. 6.20.3",
        )

        self.Q_useful = Var(self.flowsheet().config.time,initialize=1, units=pyunits.W, doc="useful net heat gain")

        self.P_pump = Var(initialize=1, units=pyunits.W, doc="pump power")

        # ==========CONSTRAINTS==========

        @self.Constraint(
            doc="corrected collector heat loss coefficient, D&B Eq. 6.20.4"
        )
        def eq_FprimeUL(b):
            return b.FprimeUL == (
                -b.mdot_test
                * b.cp_test
                / b.area_coll
                * log(1 - b.FRUL * b.area_coll / (b.mdot_test * b.cp_test))
            )

        @self.Constraint(doc="ratio of FRta_use to FRta_test, D&B Eq. 6.20.3")
        def eq_r(b):
            return b.r == (
                (
                    b.mdot_test
                    * b.ncoll
                    * b.cp_use
                    / (b.area_coll * b.ncoll)
                    * (
                        1
                        - exp(
                            -b.area_coll
                            * b.ncoll
                            * b.FprimeUL
                            / (b.mdot_test * b.ncoll * b.cp_use)
                        )
                    )
                )
                / self.FRUL
            )

        @self.Constraint(self.flowsheet().config.time,
            doc="useful net heat gain, not accounting for pipe heat losses or a heat exchanger, D&B Eq. 6.8.1"
        )
        def eq_Q_useful(b,t):
            return b.Q_useful[t] == (
                b.area_coll
                * b.ncoll
                * b.r
                * (b.FRta * b.G_trans - b.FRUL * (b.inlet_block[t].temperature -  b.outlet_block[t].temperature)) #(b.T_in[t] - b.T_amb))
            )
        
        @self.Constraint(self.flowsheet().config.time, 
                         doc='Constraint to calculate the outlet temperature from FPC')
        def outlet_temp(b,t):
            return  b.outlet_block[t].temperature == (
                # b.inlet_block[t].temperature + b.Q_useful[t]/b.mdot/b.cp_use
                b.inlet_block[t].temperature + 
                b.Q_useful[t]/b.inlet_block[t].flow_mass_phase_comp['Liq','H2O']/b.cp_use
            )

        @self.Constraint(doc="pump power")
        def eq_P_pump(b):
            return self.P_pump == (self.pump_watts / self.pump_eff)


    def initialize_build(
            self, 
            state_args=None,
            outlvl=idaeslog.NOTSET, 
            solver=None, 
            optarg=None
    ):

        # Set solver options
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="properties")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="properties")

        # Create solver
        opt = get_solver(solver=solver, options=optarg)    
                
        self.inlet_block.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,
        )

        if state_args is None:
            self.state_args = state_args = {}
            state_dict = self.inlet_block[
                self.flowsheet().config.time.first()
            ].define_port_members()

            for k in state_dict.keys():
                if state_dict[k].is_indexed():
                    state_args[k] = {}
                    for m in state_dict[k].keys():
                        state_args[k][m] = state_dict[k][m].value
                else:
                    state_args[k] = state_dict[k].value

        state_args_out = deepcopy(state_args)

        self.outlet_block.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args_out,
        )

        # solve unit
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(self, tee=slc.tee)
        init_log.info(
            "FPC initialization status {}.".format(idaeslog.condition(res))
        )

