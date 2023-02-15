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
"""
This property package computes a multi-component aqueous solution that can
contain ionic and/or neutral solute species. It supports basic calculation 
of component quanitities and some physical, chemical and electrical properties. 

This property package was formerly named as "ion_DSPMDE_prop_pack" for its use of
Donnan Steric Pore Model with Dielectric Exclusion (DSPMDE).
"""

# TODO:
#  -add calc option for Stokes radius from Stokes Einstein
#  -add viscosity as func of temp and concentration

# Import Python libraries
import idaes.logger as idaeslog

from enum import Enum, auto

# Import Pyomo libraries
from pyomo.environ import (
    Constraint,
    Expression,
    Reals,
    NonNegativeReals,
    log,
    Var,
    Param,
    Set,
    Suffix,
    value,
    check_optimal_termination,
    units as pyunits,
)
from pyomo.common.config import ConfigValue, In

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    MaterialFlowBasis,
    PhysicalParameterBlock,
    StateBlockData,
    StateBlock,
    MaterialBalanceType,
    EnergyBalanceType,
)
from idaes.core.base.components import Solute, Solvent, Cation, Anion
from idaes.core.base.phases import AqueousPhase
from idaes.core.util.constants import Constants
from idaes.core.util.initialization import (
    fix_state_vars,
    revert_state_vars,
    solve_indexed_blocks,
)
from idaes.core.util.misc import add_object_reference
from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    number_unfixed_variables,
)
from idaes.core.util.exceptions import ConfigurationError, InitializationError
import idaes.core.util.scaling as iscale
from watertap.core.util.scaling import transform_property_constraints

# Set up logger
_log = idaeslog.getLogger(__name__)


class ActivityCoefficientModel(Enum):
    ideal = auto()  # Ideal
    davies = auto()  # Davies


class DensityCalculation(Enum):
    constant = auto()  # constant @ 1000 kg/m3
    seawater = auto()  # seawater correlation for TDS from Sharqawy
    laliberte = (
        auto()
    )  # Laliberte correlation using apparent density #TODO add this later with reference


@declare_process_block_class("ChemSofteningParameterBlock")
class ChemSofteningParameterData(PhysicalParameterBlock):
    CONFIG = PhysicalParameterBlock.CONFIG()

    CONFIG.declare(
        "solute_list",
        ConfigValue(domain=list, description="List of solute species names"),
    )

    CONFIG.declare(
        "diffusivity_data",
        ConfigValue(
            default={},
            domain=dict,
            description="Dict of solute species names and bulk ion diffusivity data",
        ),
    )

    CONFIG.declare(
        "mw_data",
        ConfigValue(
            default={},
            domain=dict,
            description="Dict of component names and molecular weight data",
        ),
    )

    CONFIG.declare(
        "charge", ConfigValue(default={}, domain=dict, description="Ion charge")
    )
    CONFIG.declare(
        "activity_coefficient_model",
        ConfigValue(
            default=ActivityCoefficientModel.ideal,
            domain=In(ActivityCoefficientModel),
            description="Activity coefficient model construction flag",
            doc="""
           Options to account for activity coefficient model.

           **default** - ``ActivityCoefficientModel.ideal``

       .. csv-table::
           :header: "Configuration Options", "Description"

           "``ActivityCoefficientModel.ideal``", "Activity coefficients equal to 1 assuming ideal solution"
           "``ActivityCoefficientModel.davies``", "Activity coefficients estimated via Davies model"
       """,
        ),
    )
    CONFIG.declare(
        "density_calculation",
        ConfigValue(
            default=DensityCalculation.constant,
            domain=In(DensityCalculation),
            description="Solution density calculation construction flag",
            doc="""
           Options to account for solution density.

           **default** - ``DensityCalculation.constant``

       .. csv-table::
           :header: "Configuration Options", "Description"

           "``DensityCalculation.constant``", "Solution density assumed constant at 1000 kg/m3"
           "``DensityCalculation.seawater``", "Solution density based on correlation for seawater (TDS)"
           "``DensityCalculation.laliberte``", "Solution density based on mixing correlation from Laliberte"
       """,
        ),
    )

    def build(self):
        """
        Callable method for Block construction.
        """
        super().build()

        self._state_block_class = ChemSofteningStateBlock

        # phases
        self.Liq = AqueousPhase()

        # list to hold all species (including water)
        self.component_list = Set()

        # components
        self.H2O = Solvent()

        # blank sets
        self.cation_set = Set()
        self.anion_set = Set()
        self.solute_set = Set()
        self.ion_set = Set()
        self.hardness_set = Set()

        for j in self.config.solute_list:
            if str(j) in ["Ca_2+", "Mg_2+"]:
                self.hardness_set.add(str(j))
            if j in self.config.charge:
                if self.config.charge[j] == 0:
                    raise ConfigurationError(
                        "The charge property should not be assigned to the neutral component: {}".format(
                            j
                        )
                    )
                elif self.config.charge[j] > 0:
                    self.add_component(
                        str(j),
                        Cation(charge=self.config.charge[j], _electrolyte=True),
                    )
                    self.component_list.add(str(j))
                    self.ion_set.add(str(j))
                else:
                    self.add_component(
                        str(j),
                        Anion(charge=self.config.charge[j], _electrolyte=True),
                    )
                    self.component_list.add(str(j))
                    self.ion_set.add(str(j))
            else:
                self.add_component(str(j), Solute())

        # reference
        # Todo: enter any relevant references

        # TODO: consider turning parameters into variables for future param estimation
        # molecular weight
        # as_caco3_conv_dict = {"Ca_2+": }
        # self.as_caco3_conv = Param(self.component_list,
        #     )

        # self.mw_CaCO3 = Param(initialize=100e-3, units=pyunits.kg/pyunits.mol)
        self.ion_set.add("H2O")
        self.mw_comp = Param(
            # self.ion_set,
            self.component_list,
            mutable=True,
            default=18e-3,
            initialize=self.config.mw_data,
            units=pyunits.kg / pyunits.mol,
            doc="Molecular weight",
        )
      
        self.pKw = Param(
            initialize=14,
            units=pyunits.dimensionless,
            doc="pKw"
        )

        self.diffus_phase_comp = Param(
            self.phase_list,
            self.ion_set | self.solute_set,
            mutable=True,
            default=1e-9,
            initialize=self.config.diffusivity_data,
            units=pyunits.m**2 * pyunits.s**-1,
            doc="Bulk diffusivity of ion",
        )
        self.visc_d_phase = Param(
            self.phase_list,
            mutable=True,
            default=1e-3,
            initialize=1e-3,  # TODO:revisit- assuming ~ 1e-3 Pa*s for pure water
            units=pyunits.Pa * pyunits.s,
            doc="Fluid viscosity",
        )

        # Ion charge
        self.charge_comp = Param(
            self.ion_set,
            mutable=True,
            default=1,
            initialize=self.config.charge,
            units=pyunits.dimensionless,
            doc="Ion charge",
        )
        # Dielectric constant of water
        self.dielectric_constant = Param(
            mutable=True,
            default=80.4,
            initialize=80.4,  # todo: make a variable with parameter values for coefficients in the function of temperature
            units=pyunits.dimensionless,
            doc="Dielectric constant of water",
        )
        self.debye_huckel_b = Param(
            mutable=True,
            default=0.3,
            initialize=0.3,
            units=pyunits.kg / pyunits.mol,
            doc="Debye Huckel constant b",
        )

        # Mass density parameters, eq. 8 in Sharqawy et al. (2010)
        dens_units = pyunits.kg / pyunits.m**3
        t_inv_units = pyunits.K**-1

        self.dens_mass_param_A1 = Var(
            within=Reals,
            initialize=9.999e2,
            units=dens_units,
            doc="Mass density parameter A1",
        )
        self.dens_mass_param_A2 = Var(
            within=Reals,
            initialize=2.034e-2,
            units=dens_units * t_inv_units,
            doc="Mass density parameter A2",
        )
        self.dens_mass_param_A3 = Var(
            within=Reals,
            initialize=-6.162e-3,
            units=dens_units * t_inv_units**2,
            doc="Mass density parameter A3",
        )
        self.dens_mass_param_A4 = Var(
            within=Reals,
            initialize=2.261e-5,
            units=dens_units * t_inv_units**3,
            doc="Mass density parameter A4",
        )
        self.dens_mass_param_A5 = Var(
            within=Reals,
            initialize=-4.657e-8,
            units=dens_units * t_inv_units**4,
            doc="Mass density parameter A5",
        )
        self.dens_mass_param_B1 = Var(
            within=Reals,
            initialize=8.020e2,
            units=dens_units,
            doc="Mass density parameter B1",
        )
        self.dens_mass_param_B2 = Var(
            within=Reals,
            initialize=-2.001,
            units=dens_units * t_inv_units,
            doc="Mass density parameter B2",
        )
        self.dens_mass_param_B3 = Var(
            within=Reals,
            initialize=1.677e-2,
            units=dens_units * t_inv_units**2,
            doc="Mass density parameter B3",
        )
        self.dens_mass_param_B4 = Var(
            within=Reals,
            initialize=-3.060e-5,
            units=dens_units * t_inv_units**3,
            doc="Mass density parameter B4",
        )
        self.dens_mass_param_B5 = Var(
            within=Reals,
            initialize=-1.613e-5,
            units=dens_units * t_inv_units**2,
            doc="Mass density parameter B5",
        )

        # traditional parameters are the only Vars currently on the block and should be fixed
        for v in self.component_objects(Var):
            v.fix()

        # ---default scaling---
        self.set_default_scaling("temperature", 1e-2)
        self.set_default_scaling("mass_frac_phase_comp", 1e2)
        self.set_default_scaling("pressure", 1e-4)
        self.set_default_scaling("dens_mass_phase", 1e-3, index="Liq")
        self.set_default_scaling("visc_d_phase", 1e3, index="Liq")
        self.set_default_scaling("diffus_phase_comp", 1e10, index="Liq")
        self.set_default_scaling("visc_k_phase", 1e6, index="Liq")

    @classmethod
    def define_metadata(cls, obj):
        """Define properties supported and units."""
        obj.add_properties(
            {
                "flow_mol_phase_comp": {"method": "_flow_mol_phase_comp"},
                "temperature": {"method": None},
                "pressure": {"method": None},
                "alkalinity": {"method": None},
                "pH": {"method": None},
                "pOH": {"method": "_pOH"},
                "pKw": {"method": "_pKw"},
                "flow_mass_phase_comp": {"method": None},
                "flow_equiv_phase_comp": {"method": "_flow_equiv_phase_comp"},
                "conc_equiv_phase_comp": {"method": "_conc_equiv_phase_comp"},
                "mass_frac_phase_comp": {"method": "_mass_frac_phase_comp"},
                "dens_mass_phase": {"method": "_dens_mass_phase"},
                "dens_mass_solvent": {"method": "_dens_mass_solvent"},
#                "flow_vol": {"method": "_flow_vol"},
                "flow_vol_phase": {"method": "_flow_vol_phase"},
                "conc_mol_phase_comp": {"method": "_conc_mol_phase_comp"},
                "conc_mass_phase_comp": {"method": "_conc_mass_phase_comp"},
                "mole_frac_phase_comp": {"method": "_mole_frac_phase_comp"},
                "molality_phase_comp": {"method": "_molality_phase_comp"},
                "diffus_phase_comp": {"method": "_diffus_phase_comp"},
                "visc_d_phase": {"method": "_visc_d_phase"},
                "visc_k_phase": {"method": "_visc_k_phase"},
                "pressure_osm_phase": {"method": "_pressure_osm_phase"},
                # "radius_stokes_comp": {"method": "_radius_stokes_comp"},
                "mw_comp": {"method": "_mw_comp"},
                "conc_mass_caco3_comp": {"method": "_conc_mass_caco3_comp"},
                # "non_carbonate_hardness_comp": {"method": None},
                # "carbonate_hardness_comp": {"method": None},
                "total_hardness": {"method": "_total_hardness"},
                "charge_comp": {"method": "_charge_comp"},

            }
        )

        obj.add_default_units(
            {
                "time": pyunits.s,
                "length": pyunits.m,
                "mass": pyunits.kg,
                "amount": pyunits.mol,
                "temperature": pyunits.K,
            }
        )


class _ChemSofteningStateBlock(StateBlock):
    """
    This Class contains methods which should be applied to Property Blocks as a whole, rather
    than individual elements of indexed Property Blocks.
    """

    def initialize(
        self,
        state_args=None,
        state_vars_fixed=False,
        hold_state=False,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
    ):
        """
        Initialization routine for property package.

        Keyword Arguments:
            state_args : Dictionary with initial guesses for the state vars
                         chosen. Note that if this method is triggered through the control
                         volume, and if initial guesses were not provided at the unit model
                         level, the control volume passes the inlet values as initial guess.
                         The keys for the state_args dictionary are:
                         flow_mol_phase_comp : value to initialize phase component flows;
                         pressure : value at which to initialize pressure;
                         temperature : value at which to initialize temperature.
            outlvl : sets output level of initialization routine (default=idaeslog.NOTSET)
            optarg : solver options dictionary object (default=None)
            state_vars_fixed : Flag to denote if state vars have already
                               been fixed.
                               - True - states have already been fixed by the control volume
                               1D. Control volume 0D does not fix the state vars, so will be
                               False if this state block is used with 0D blocks.
                               - False - states have not been fixed. The state block will deal
                               with fixing/unfixing.
            solver : Solver object to use during initialization. If None
                     is provided, it will use the default solver for IDAES (default = None)
            hold_state : flag indicating whether the initialization routine
                         should unfix any state variables fixed during
                         initialization (default=False).
                         - True - state variables are not unfixed, and a dict of returned
                         containing flags for which states were fixed during initialization.
                         - False - state variables are unfixed after initialization by calling
                         the release_state method.

        Returns:
            If hold_states is True, returns a dict containing flags for which states were fixed
            during initialization.
        """
        # Get loggers
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="properties")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="properties")

        # Set solver and options
        opt = get_solver(solver, optarg)

        # Fix state variables
        flags = fix_state_vars(self, state_args)

        # initialize vars caculated from state vars
        for k in self.keys():

            # Vars indexed by phase and component_list
            for j in self[k].params.component_list:
                if self[k].is_property_constructed("mass_frac_phase_comp"):
                    self[k].mass_frac_phase_comp["Liq", j].set_value(
                        self[k].flow_mass_phase_comp["Liq", j]
                        / sum(
                            self[k].flow_mass_phase_comp["Liq", j]
                            for j in self[k].params.component_list
                        )
                    )
                if self[k].is_property_constructed("conc_mass_phase_comp"):
                    self[k].conc_mass_phase_comp["Liq", j].set_value(
                        self[k].dens_mass_phase["Liq"]
                        * self[k].mass_frac_phase_comp["Liq", j]
                    )

                if self[k].is_property_constructed("conc_mol_phase_comp"):
                    self[k].conc_mol_phase_comp["Liq", j].set_value(
                        self[k].conc_mass_phase_comp["Liq", j]
                        / self[k].params.mw_comp[j]
                    )

                # if self[k].is_property_constructed("flow_mass_phase_comp"):
                #     self[k].flow_mass_phase_comp["Liq", j].set_value(
                #         self[k].flow_mol_phase_comp["Liq", j]
                #         * self[k].params.mw_comp[j]
                #     )
                if self[k].is_property_constructed("mole_frac_phase_comp"):
                    self[k].mole_frac_phase_comp["Liq", j].set_value(
                        self[k].flow_mol_phase_comp["Liq", j]
                        / sum(
                            self[k].flow_mol_phase_comp["Liq", j]
                            for j in self[k].params.component_list
                        )
                    )

            # Vars indexed by ion_set | solute_set
            for j in self[k].params.ion_set | self[k].params.solute_set:
                if self[k].is_property_constructed("molality_phase_comp"):
                    self[k].molality_phase_comp["Liq", j].set_value(
                        self[k].flow_mol_phase_comp["Liq", j]
                        / self[k].flow_mol_phase_comp["Liq", "H2O"]
                        / self[k].params.mw_comp["H2O"]
                    )

            # Vars indexed by ion_set
            for j in self[k].params.ion_set:
                # if (
                #     self[k].is_property_constructed("elec_mobility_phase_comp")
                #     and self[k].params.config.elec_mobility_calculation
                #     == ElectricalMobilityCalculation.EinsteinRelation
                # ):
                #     self[k].elec_mobility_phase_comp["Liq", j].set_value(
                #         self[k].diffus_phase_comp["Liq", j]
                #         * abs(self[k].charge_comp[j])
                #         * Constants.faraday_constant
                #         / (Constants.gas_constant * self[k].temperature)
                #     )
                if self[k].is_property_constructed("conc_equiv_phase_comp"):
                    self[k].conc_equiv_phase_comp["Liq", j].set_value(
                        self[k].conc_mol_phase_comp["Liq", j]
                        / abs(self[k].params.charge_comp[j])
                    )
                if self[k].is_property_constructed("flow_equiv_phase_comp"):
                    self[k].flow_equiv_phase_comp["Liq", j].set_value(
                        self[k].flow_mol_phase_comp["Liq", j]
                        * abs(self[k].params.charge_comp[j])
                    )
            # Vars not indexed or indexed only by phase
            if self[k].is_property_constructed("flow_vol_phase"):
                self[k].flow_vol_phase["Liq"].set_value(
                    sum(
                        self[k].flow_mass_phase_comp["Liq", j]
                        for j in self[k].params.component_list
                    )
                    / self[k].dens_mass_phase["Liq"]
                )
            if self[k].is_property_constructed("visc_k_phase"):
                self[k].visc_k_phase["Liq"].set_value(
                    self[k].visc_d_phase["Liq"] / self[k].dens_mass_phase["Liq"]
                )
            # if self[k].is_property_constructed("ionic_strength_molal"):
            #     self[k].ionic_strength_molal.set_value(
            #         0.5
            #         * sum(
            #             self[k].charge_comp[j] ** 2
            #             * self[k].molality_phase_comp["Liq", j]
            #             for j in self[k].params.ion_set | self[k].params.solute_set
            #         )
            #     )
            # if self[k].is_property_constructed("pressure_osm_phase"):
            #     self[k].pressure_osm_phase["Liq"].set_value(
            #         sum(
            #             self[k].conc_mol_phase_comp["Liq", j]
            #             for j in self[k].params.ion_set | self[k].params.solute_set
            #         )
            #         * Constants.gas_constant
            #         * self[k].temperature
            #     )

            # if (
            #     self[k].is_property_constructed("equiv_conductivity_phase")
            #     and self[k].params.config.equiv_conductivity_calculation
            #     == EquivalentConductivityCalculation.ElectricalMobility
            # ):
            #     self[k].equiv_conductivity_phase["Liq"].set_value(
            #         sum(
            #             Constants.faraday_constant
            #             * abs(self[k].charge_comp[j])
            #             * self[k].elec_mobility_phase_comp["Liq", j]
            #             * self[k].conc_mol_phase_comp["Liq", j]
            #             for j in self[k].params.ion_set
            #         )
            #         / sum(
            #             abs(self[k].charge_comp[j])
            #             * self[k].conc_mol_phase_comp["Liq", j]
            #             for j in self[k].params.cation_set
            #         )
            #     )
            # if self[k].is_property_constructed("elec_cond_phase"):
            #     self[k].elec_cond_phase["Liq"].set_value(
            #         self[k].equiv_conductivity_phase["Liq"]
            #         * sum(
            #             abs(self[k].charge_comp[j])
            #             * self[k].conc_mol_phase_comp["Liq", j]
            #             for j in self[k].params.cation_set
            #         )
            #     )

        # Check when the state vars are fixed already result in dof 0
        for k in self.keys():
            dof = degrees_of_freedom(self[k])
            if dof != 0:
                raise InitializationError(
                    "\nWhile initializing {sb_name}, the degrees of freedom "
                    "are {dof}, when zero is required. \nInitialization assumes "
                    "that the state variables should be fixed and that no other "
                    "variables are fixed. \nIf other properties have a "
                    "predetermined value, use the calculate_state method "
                    "before using initialize to determine the values for "
                    "the state variables and avoid fixing the property variables."
                    "".format(sb_name=self.name, dof=dof)
                )

        # ---------------------------------------------------------------------
        skip_solve = True  # skip solve if only state variables are present
        for k in self.keys():
            if number_unfixed_variables(self[k]) != 0:

                skip_solve = False

        if not skip_solve:
            # Initialize properties
            with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                results = solve_indexed_blocks(opt, [self], tee=slc.tee)
                if not check_optimal_termination(results):
                    raise InitializationError(
                        "The property package failed to solve during initialization."
                    )
            init_log.info_high(
                "Property initialization: {}.".format(idaeslog.condition(results))
            )

        # ---------------------------------------------------------------------
        # If input block, return flags, else release state
        if state_vars_fixed is False:
            if hold_state is True:
                return flags
            else:
                self.release_state(flags)

    def release_state(self, flags, outlvl=idaeslog.NOTSET):
        """
        Method to release state variables fixed during initialisation.

        Keyword Arguments:
            flags : dict containing information of which state variables
                    were fixed during initialization, and should now be
                    unfixed. This dict is returned by initialize if
                    hold_state=True.
            outlvl : sets output level of logging
        """
        # Unfix state variables
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="properties")
        revert_state_vars(self, flags)
        init_log.info_high("{} State Released.".format(self.name))

    def calculate_state(
        self,
        var_args=None,
        hold_state=False,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
    ):
        """
        Solves state blocks given a set of variables and their values. These variables can be
        state variables or properties. This method is typically used before initialization to
        solve for state variables because non-state variables (i.e. properties) cannot be fixed
        in initialization routines.

        Keyword Arguments:
            var_args : dictionary with variables and their values, they
                       can be state variables or properties
                       {(VAR_NAME, INDEX): VALUE}
            hold_state : flag indicating whether all of the state
                         variables should be fixed after calculate state.
                         True - State variables will be fixed.
                         False - State variables will remain unfixed, unless already fixed.
            outlvl : idaes logger object that sets output level of solve
                     call (default=idaeslog.NOTSET)
            solver : solver name string if None is provided the default
                     solver for IDAES will be used (default = None)
            optarg : solver options dictionary object (default={})

        Returns:
            results object from state block solve
        """
        # Get logger
        solve_log = idaeslog.getSolveLogger(self.name, level=outlvl, tag="properties")

        # Initialize at current state values (not user provided)
        self.initialize(solver=solver, optarg=optarg, outlvl=outlvl)

        # Set solver and options
        opt = get_solver(solver, optarg)

        # Fix variables and check degrees of freedom
        flags = (
            {}
        )  # dictionary noting which variables were fixed and their previous state
        for k in self.keys():
            sb = self[k]
            for (v_name, ind), val in var_args.items():
                var = getattr(sb, v_name)
                if iscale.get_scaling_factor(var[ind]) is None:
                    _log.warning(
                        "While using the calculate_state method on {sb_name}, variable {v_name} "
                        "was provided as an argument in var_args, but it does not have a scaling "
                        "factor. This suggests that the calculate_scaling_factor method has not been "
                        "used or the variable was created on demand after the scaling factors were "
                        "calculated. It is recommended to touch all relevant variables (i.e. call "
                        "them or set an initial value) before using the calculate_scaling_factor "
                        "method.".format(v_name=v_name, sb_name=sb.name)
                    )
                if var[ind].is_fixed():
                    flags[(k, v_name, ind)] = True
                    if value(var[ind]) != val:
                        raise ConfigurationError(
                            "While using the calculate_state method on {sb_name}, {v_name} was "
                            "fixed to a value {val}, but it was already fixed to value {val_2}. "
                            "Unfix the variable before calling the calculate_state "
                            "method or update var_args."
                            "".format(
                                sb_name=sb.name,
                                v_name=var.name,
                                val=val,
                                val_2=value(var[ind]),
                            )
                        )
                else:
                    flags[(k, v_name, ind)] = False
                    var[ind].fix(val)

            if degrees_of_freedom(sb) != 0:
                raise RuntimeError(
                    "While using the calculate_state method on {sb_name}, the degrees "
                    "of freedom were {dof}, but 0 is required. Check var_args and ensure "
                    "the correct fixed variables are provided."
                    "".format(sb_name=sb.name, dof=degrees_of_freedom(sb))
                )

        # Solve
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            results = solve_indexed_blocks(opt, [self], tee=slc.tee)
            solve_log.info_high(
                "Calculate state: {}.".format(idaeslog.condition(results))
            )

        if not check_optimal_termination(results):
            _log.warning(
                "While using the calculate_state method on {sb_name}, the solver failed "
                "to converge to an optimal solution. This suggests that the user provided "
                "infeasible inputs, or that the model is poorly scaled, poorly initialized, "
                "or degenerate."
            )

        # unfix all variables fixed with var_args
        for (k, v_name, ind), previously_fixed in flags.items():
            if not previously_fixed:
                var = getattr(self[k], v_name)
                var[ind].unfix()

        # fix state variables if hold_state
        if hold_state:
            fix_state_vars(self)

        return results


@declare_process_block_class("ChemSofteningStateBlock", block_class=_ChemSofteningStateBlock)
class ChemSofteningStateBlockData(StateBlockData):
    def build(self):
        """Callable method for Block construction."""
        super().build()

        self.scaling_factor = Suffix(direction=Suffix.EXPORT)

        # Add state variables
        self.flow_mass_phase_comp = Var(
            self.params.phase_list,
            self.params.component_list,
            initialize=0.1,  # todo: revisit
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.kg / pyunits.s,
            doc="Mole flow rate",
        )

        self.temperature = Var(
            initialize=298.15,
            bounds=(273.15, 373.15),
            domain=NonNegativeReals,
            units=pyunits.K,
            doc="State temperature",
        )

        self.pressure = Var(
            initialize=101325,
            bounds=(1e5, None),
            domain=NonNegativeReals,
            units=pyunits.Pa,
            doc="State pressure",
        )

        self.pH = Var(
            initialize=7,
            bounds=(0, 14),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="State pH",
        )

        self.alkalinity = Var(
            initialize=100,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.kg / pyunits.m**3,
            doc="State alkalinity",
        )

        self.carbonate_hardness_comp = Var(
            self.params.hardness_set,
            initialize=100,
            bounds=(0, None),
            units=pyunits.kg / pyunits.m**3,
            doc="Carbonate hardness in CaCO3 equivalents",
        )

        self.non_carbonate_hardness_comp = Var(
            self.params.hardness_set,
            initialize=100,
            bounds=(0, None),
            units=pyunits.kg / pyunits.m**3,
            doc="Carbonate hardness in CaCO3 equivalents",
        )


    # -----------------------------------------------------------------------------
    # Property Methods
    def _pOH(self):
        self.pOH = Var(
            initialize=7,
            bounds=(0, 16),
            units=pyunits.dimensionless,
            doc="pOH",
        )
    
        def rule_pOH(b):
            return b.pOH == b.pKw - b.pH
        
        self.eq_pOH = Constraint(rule=rule_pOH)


    def _mass_frac_phase_comp(self):
        self.mass_frac_phase_comp = Var(
            self.params.phase_list,
            self.params.component_list,
            initialize=0.5,
            bounds=(0, 1.001),
            units=pyunits.kg / pyunits.kg,
            doc="Mass fraction",
        )

        def rule_mass_frac_phase_comp(b, p, j):
            return b.mass_frac_phase_comp[p, j] == b.flow_mass_phase_comp[p, j] / sum(
                b.flow_mass_phase_comp[p, j] for j in self.params.component_list
            )

        self.eq_mass_frac_phase_comp = Constraint(
            self.params.phase_list,
            self.params.component_list,
            rule=rule_mass_frac_phase_comp,
        )

    def _conc_mass_caco3_comp(self):
        self.conc_mass_caco3_comp = Var(
            self.params.hardness_set,
            initialize=100,
            bounds=(0, None),
            units=pyunits.kg / pyunits.m**3,
            doc="Mass concentration in CaCO3 equivalents",
        )

        self.equivalent_wt_caco3 = Param(
            initialize=0.050, 
            units=pyunits.kg/pyunits.mol, 
            doc="Equivalent weight of CaCO3"
            )

        # self.caco3_conversion_co2 = Param(
        #     initialize=1612, 
        #     units=pyunits.dimensionless, 
        #     doc="Equivalent weight of CaCO3"
        #     ) 

        def rule_conc_mass_caco3_comp(b, j):
            return b.conc_mass_caco3_comp[j] == b.equivalent_wt_caco3 / (b.params.mw_comp[j] / abs(b.params.charge_comp[j])) * b.conc_mass_phase_comp["Liq", j]

        self.eq_conc_mass_caco3_comp = Constraint(
            self.params.hardness_set,
            rule=rule_conc_mass_caco3_comp,
        )

    # def _carbonate_hardness_comp(self):
    #     self.carbonate_hardness_comp = Var(
    #         self.params.hardness_set,
    #         initialize=100,
    #         bounds=(0, None),
    #         units=pyunits.kg / pyunits.m**3,
    #         doc="Carbonate hardness in CaCO3 equivalents",
    #     )

    #     def rule_carbonate_hardness_comp(b, j):
    #         if value(b.conc_mass_caco3_comp["Ca_2+"]) >= value(b.alkalinity):
    #             if j == "Ca_2+":
    #                 b.carbonate_hardness_comp[j].fix(value(b.alkalinity))
    #                 return Constraint.Skip
    #             if j == "Mg_2+":
    #                 b.carbonate_hardness_comp[j].fix(0)
    #                 return Constraint.Skip
    #         else:
    #             if j == "Ca_2+":
    #                 b.carbonate_hardness_comp[j].fix(value(b.conc_mass_caco3_comp["Ca_2+"]))
    #                 return Constraint.Skip
    #             if j == "Mg_2+":
    #                 b.carbonate_hardness_comp[j].fix(value(b.alkalinity - b.conc_mass_caco3_comp["Ca_2+"]))
    #                 return Constraint.Skip

    #     self.eq_carbonate_hardness_comp = Constraint(
    #         self.params.hardness_set,
    #         rule=rule_carbonate_hardness_comp,
    #     )

    # def _non_carbonate_hardness_comp(self):
    #     self.non_carbonate_hardness_comp = Var(
    #         self.params.hardness_set,
    #         initialize=100,
    #         bounds=(0, None),
    #         units=pyunits.kg / pyunits.m**3,
    #         doc="Non-carbonate hardness in CaCO3 equivalents",
    #     )

    #     def rule_non_carbonate_hardness_comp(b, j):
    #         if value(b.conc_mass_caco3_comp["Ca_2+"]) >= value(b.alkalinity):
    #             if j == "Ca_2+":
    #                 b.non_carbonate_hardness_comp[j].fix(value(b.conc_mass_caco3_comp["Ca_2+"] - b.alkalinity))
    #                 return Constraint.Skip
    #             if j == "Mg_2+":
    #                 b.non_carbonate_hardness_comp[j].fix(value(b.conc_mass_caco3_comp["Mg_2+"]))
    #                 return Constraint.Skip
                    
    #         else:
    #             if j == "Ca_2+":
    #                 b.non_carbonate_hardness_comp[j].fix(0)
    #                 return Constraint.Skip
    #             if j == "Mg_2+":
    #                 TH = sum(b.conc_mass_caco3_comp[p] for p in b.params.hardness_set)
    #                 b.non_carbonate_hardness_comp[j].fix(value(TH - b.alkalinity))
    #                 return Constraint.Skip

    #     self.eq_non_carbonate_hardness_comp = Constraint(
    #         self.params.hardness_set,
    #         rule=rule_non_carbonate_hardness_comp,
    #     )

    def _total_hardness(self):
        self.total_hardness = Var(
            initialize=1e3,
            bounds=(5e2, 2e3),
            units=pyunits.kg * pyunits.m**-3,
            doc="Mass density",
        )
    def _total_hardness(self):
        def rule_total_hardness(b):
            return sum(b.conc_mass_caco3_comp[p] for p in b.params.hardness_set)

        self.total_hardness = Expression(rule=rule_total_hardness)


    def _dens_mass_phase(self):
        self.dens_mass_phase = Var(
            ["Liq"],
            initialize=1e3,
            bounds=(5e2, 2e3),
            units=pyunits.kg * pyunits.m**-3,
            doc="Mass density",
        )
        # TODO: reconsider this approach for solution density based on arbitrary solute_list
        def rule_dens_mass_phase(b, p):
            if b.params.config.density_calculation == DensityCalculation.constant:
                return b.dens_mass_phase[p] == 1000 * pyunits.kg * pyunits.m**-3
                # b.dens_mass_phase[p].fix(1000 * pyunits.kg * pyunits.m**-3)
                # return Constraint.Skip
            elif b.params.config.density_calculation == DensityCalculation.seawater:
                # density, eq. 8 in Sharqawy #TODO- add Sharqawy reference
                t = b.temperature - 273.15 * pyunits.K
                s = sum(
                    b.mass_frac_phase_comp[p, j]
                    for j in b.params.ion_set | b.params.solute_set
                )
                dens_mass = (
                    b.dens_mass_solvent
                    + b.params.dens_mass_param_B1 * s
                    + b.params.dens_mass_param_B2 * s * t
                    + b.params.dens_mass_param_B3 * s * t**2
                    + b.params.dens_mass_param_B4 * s * t**3
                    + b.params.dens_mass_param_B5 * s**2 * t**2
                )
                return b.dens_mass_phase[p] == dens_mass

        self.eq_dens_mass_phase = Constraint(["Liq"], rule=rule_dens_mass_phase)

    def _dens_mass_solvent(self):
        self.dens_mass_solvent = Var(
            initialize=1e3,
            bounds=(1, 1e6),
            units=pyunits.kg * pyunits.m**-3,
            doc="Mass density of pure water",
        )

        def rule_dens_mass_solvent(b):  # density, eq. 8 in Sharqawy
            t = b.temperature - 273.15 * pyunits.K
            dens_mass_w = (
                b.params.dens_mass_param_A1
                + b.params.dens_mass_param_A2 * t
                + b.params.dens_mass_param_A3 * t**2
                + b.params.dens_mass_param_A4 * t**3
                + b.params.dens_mass_param_A5 * t**4
            )
            return b.dens_mass_solvent == dens_mass_w

        self.eq_dens_mass_solvent = Constraint(rule=rule_dens_mass_solvent)

    def _flow_vol_phase(self):
        self.flow_vol_phase = Var(
            self.params.phase_list,
            initialize=0.001,
            bounds=(0, None),
            units=pyunits.m**3 / pyunits.s,
            doc="Volumetric flow rate",
        )

        def rule_flow_vol_phase(b, p):
            return (
                b.flow_vol_phase[p]
                == sum(
                    b.flow_mass_phase_comp[p, j]
                    for j in self.params.component_list
                )
                / b.dens_mass_phase[p]
            )

        self.eq_flow_vol_phase = Constraint(
            self.params.phase_list, rule=rule_flow_vol_phase
        )

    def _flow_vol(self):
        def rule_flow_vol(b):
            return sum(b.flow_vol_phase[p] for p in self.params.phase_list)

        self.flow_vol = Expression(rule=rule_flow_vol)

    def _conc_mol_phase_comp(self):
        self.conc_mol_phase_comp = Var(
            self.params.phase_list,
            self.params.component_list,
            initialize=500,
            bounds=(0, None),
            units=pyunits.mol * pyunits.m**-3,
            doc="Molar concentration",
        )

        def rule_conc_mol_phase_comp(b, p, j):
            return (
                b.conc_mol_phase_comp[p, j] * b.params.mw_comp[j]
                == b.conc_mass_phase_comp[p, j]
            )

        self.eq_conc_mol_phase_comp = Constraint(
            self.params.phase_list,
            self.params.component_list,
            rule=rule_conc_mol_phase_comp,
        )

    def _conc_mass_phase_comp(self):
        self.conc_mass_phase_comp = Var(
            self.params.phase_list,
            self.params.component_list,
            initialize=10,
            bounds=(0, 2e3),
            units=pyunits.kg * pyunits.m**-3,
            doc="Mass concentration",
        )

        def rule_conc_mass_phase_comp(b, p, j):
            return (
                b.conc_mass_phase_comp[p, j]
                == b.dens_mass_phase[p] * b.mass_frac_phase_comp[p, j]
            )

        self.eq_conc_mass_phase_comp = Constraint(
            self.params.phase_list,
            self.params.component_list,
            rule=rule_conc_mass_phase_comp,
        )
    def _flow_mol_phase_comp(self):
        self.flow_mol_phase_comp = Var(
            self.phase_component_set,
            initialize=100,
            bounds=(None, None),
            domain=NonNegativeReals,
            units=pyunits.mol / pyunits.s,
            doc="Molar flowrate",
        )

        def rule_flow_mol_phase_comp(b, p, j):
            return (
                b.flow_mol_phase_comp[p, j]
                == b.flow_mass_phase_comp[p, j] / b.params.mw_comp[j]
            )

        self.eq_flow_mol_phase_comp = Constraint(
            self.phase_component_set, rule=rule_flow_mol_phase_comp
        )
    # def _flow_mass_phase_comp(self):
    #     self.flow_mass_phase_comp = Var(
    #         self.params.phase_list,
    #         self.params.component_list,
    #         initialize=0.5,
    #         bounds=(0, None),
    #         units=pyunits.kg / pyunits.s,
    #         doc="Component Mass flowrate",
    #     )

    #     def rule_flow_mass_phase_comp(b, p, j):
    #         return (
    #             b.flow_mass_phase_comp[p, j]
    #             == b.flow_mol_phase_comp[p, j] * b.params.mw_comp[j]
    #         )

    #     self.eq_flow_mass_phase_comp = Constraint(
    #         self.params.phase_list,
    #         self.params.component_list,
    #         rule=rule_flow_mass_phase_comp,
    #     )

    def _flow_equiv_phase_comp(self):
        self.flow_equiv_phase_comp = Var(
            self.params.phase_list,
            self.params.ion_set,
            initialize=0.1,
            bounds=(0, None),
            units=pyunits.mol / pyunits.s,
            doc="Component equivalent charge flowrate",
        )

        def rule_flow_equiv_phase_comp(b, p, j):
            return b.flow_equiv_phase_comp[p, j] == b.flow_mol_phase_comp[p, j] * abs(
                b.params.charge_comp[j]
            )

        self.eq_flow_equiv_phase_comp = Constraint(
            self.params.phase_list,
            self.params.ion_set,
            rule=rule_flow_equiv_phase_comp,
        )

    def _conc_equiv_phase_comp(self):
        self.conc_equiv_phase_comp = Var(
            self.params.phase_list,
            self.params.ion_set,
            initialize=500,
            bounds=(0, None),
            units=pyunits.mol / pyunits.m**3,
            doc="Equivalent charge concentration",
        )

        def rule_conc_equiv_phase_comp(b, p, j):

            return b.conc_equiv_phase_comp[p, j] == b.conc_mol_phase_comp[p, j] * abs(
                b.params.charge_comp[j]
            )

        self.eq_conc_equiv_phase_comp = Constraint(
            self.params.phase_list,
            self.params.ion_set,
            rule=rule_conc_equiv_phase_comp,
        )

    def _mole_frac_phase_comp(self):
        self.mole_frac_phase_comp = Var(
            self.params.phase_list,
            self.params.component_list,
            initialize=0.5,
            bounds=(0, 1.001),
            units=pyunits.dimensionless,
            doc="Mole fraction",
        )

        def rule_mole_frac_phase_comp(b, p, j):
            return b.mole_frac_phase_comp[p, j] == b.flow_mol_phase_comp[p, j] / sum(
                b.flow_mol_phase_comp[p, j] for j in b.params.component_list
            )

        self.eq_mole_frac_phase_comp = Constraint(
            self.params.phase_list,
            self.params.component_list,
            rule=rule_mole_frac_phase_comp,
        )

    def _molality_phase_comp(self):
        self.molality_phase_comp = Var(
            self.params.phase_list,
            self.params.ion_set | self.params.solute_set,
            initialize=1,
            bounds=(0, 10),
            units=pyunits.mole / pyunits.kg,
            doc="Molality",
        )

        def rule_molality_phase_comp(b, p, j):
            return (
                b.molality_phase_comp[p, j]
                == b.flow_mol_phase_comp[p, j]
                / b.flow_mol_phase_comp[p, "H2O"]
                / b.params.mw_comp["H2O"]
            )

        self.eq_molality_phase_comp = Constraint(
            self.params.phase_list,
            self.params.ion_set | self.params.solute_set,
            rule=rule_molality_phase_comp,
        )

    def _visc_k_phase(self):
        self.visc_k_phase = Var(
            ["Liq"],
            initialize=1e-6,
            bounds=(9e-7, 5e-2),
            units=pyunits.m**2 / pyunits.s,
            doc="Kinematic Viscosity",
        )

        def rule_visc_k_phase(b, p):
            return b.visc_d_phase[p] == b.visc_k_phase[p] * b.dens_mass_phase[p]

        self.eq_visc_k_phase = Constraint(["Liq"], rule=rule_visc_k_phase)

    # def _radius_stokes_comp(self):
    #     add_object_reference(self, "radius_stokes_comp", self.params.radius_stokes_comp)

    def _pKw(self):

        self.pKw = Var(
            initialize=14,
            # domain=(10, 16),
            units=pyunits.dimensionless,
            doc="pKw"
        )

        self.pKw_coeff_A = Param(
            initialize=4470.99
        )

        self.pKw_coeff_B = Param(
            initialize=0.017060
        )

        self.pKw_coeff_C = Param(
            initialize=6.0875
        )

        def rule_pKw(b):
            return b.pKw == b.pKw_coeff_A / b.temperature + b.pKw_coeff_B * b.temperature - b.pKw_coeff_C
        
        self.eq_pKw = Constraint(rule=rule_pKw)
        # add_object_reference(self, "pKw", self.params.pKw)

    def _diffus_phase_comp(self):
        add_object_reference(self, "diffus_phase_comp", self.params.diffus_phase_comp)

    def _visc_d_phase(self):
        add_object_reference(self, "visc_d_phase", self.params.visc_d_phase)

    def _mw_comp(self):
        add_object_reference(self, "mw_comp", self.params.mw_comp)

  

    def _charge_comp(self):
        add_object_reference(self, "charge_comp", self.params.charge_comp)


    def _act_coeff_phase_comp(self):
        self.act_coeff_phase_comp = Var(
            self.params.phase_list,
            self.params.ion_set | self.params.solute_set,
            initialize=0.7,
            domain=NonNegativeReals,
            bounds=(0, 1.001),
            units=pyunits.dimensionless,
            doc="activity coefficient of component",
        )

        def rule_act_coeff_phase_comp(b, p, j):
            if (
                b.params.config.activity_coefficient_model
                == ActivityCoefficientModel.ideal
            ):
                return b.act_coeff_phase_comp[p, j] == 1
            elif (
                b.params.config.activity_coefficient_model
                == ActivityCoefficientModel.davies
            ):
                I = b.ionic_strength_molal
                return log(
                    b.act_coeff_phase_comp[p, j]
                ) == -b.debye_huckel_constant * b.charge_comp[j] ** 2 * (
                    I**0.5 / (1 * pyunits.mole**0.5 / pyunits.kg**0.5 + I**0.5)
                    - b.params.debye_huckel_b * I
                )

        self.eq_act_coeff_phase_comp = Constraint(
            self.params.phase_list,
            self.params.ion_set | self.params.solute_set,
            rule=rule_act_coeff_phase_comp,
        )

    # TODO: note- assuming molal ionic strength goes into Debye Huckel relationship;
    # the MIT's DSPMDE paper indicates usage of molar concentration
    def _ionic_strength_molal(self):
        self.ionic_strength_molal = Var(
            initialize=1,
            domain=NonNegativeReals,
            units=pyunits.mol / pyunits.kg,
            doc="Molal ionic strength",
        )

        def rule_ionic_strength_molal(b):
            return b.ionic_strength_molal == 0.5 * sum(
                b.charge_comp[j] ** 2 * b.molality_phase_comp["Liq", j]
                for j in self.params.ion_set
            )

        self.eq_ionic_strength_molal = Constraint(rule=rule_ionic_strength_molal)

    def _debye_huckel_constant(self):
        self.debye_huckel_constant = Var(
            initialize=1,
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            # TODO: units are technically (kg/mol)**0.5, but Debye Huckel equation
            #  is empirical and units don't seem to cancel as typical. leaving as dimensionless for now
            doc="Temperature-dependent Debye Huckel constant A",
        )

        def rule_debye_huckel_constant(b):
            return (
                b.debye_huckel_constant
                == ((2 * Constants.pi * Constants.avogadro_number) ** 0.5 / log(10))
                * (
                    Constants.elemental_charge**2
                    / (
                        4
                        * Constants.pi
                        * Constants.vacuum_electric_permittivity
                        * b.params.dielectric_constant
                        * Constants.boltzmann_constant
                        * b.temperature
                    )
                )
                ** (3 / 2)
                * (
                    pyunits.coulomb**3
                    * pyunits.m**1.5
                    / pyunits.farad**1.5
                    / pyunits.J**1.5
                    / pyunits.mol**0.5
                )
                ** -1
            )

        self.eq_debye_huckel_constant = Constraint(rule=rule_debye_huckel_constant)

    # TODO: change osmotic pressure calc
    def _pressure_osm_phase(self):
        self.pressure_osm_phase = Var(
            self.params.phase_list,
            initialize=1e6,
            bounds=(0, None),
            units=pyunits.Pa,
            doc="van't Hoff Osmotic pressure",
        )

        def rule_pressure_osm_phase(b, p):
            return (
                b.pressure_osm_phase[p]
                == sum(
                    b.conc_mol_phase_comp[p, j]
                    for j in self.params.ion_set | self.params.solute_set
                )
                * Constants.gas_constant
                * b.temperature
            )

        self.eq_pressure_osm_phase = Constraint(
            self.params.phase_list, rule=rule_pressure_osm_phase
        )

    def _trans_num_phase_comp(self):
        self.trans_num_phase_comp = Var(
            self.params.phase_list,
            self.params.ion_set,
            initialize=0.5,
            units=pyunits.dimensionless,
            doc="Ion transport number in the liquid phase",
        )

        def rule_trans_num_phase_comp(b, p, j):
            if (
                self.params.config.trans_num_calculation
                == TransportNumberCalculation.none
            ):
                if (p, j) not in self.params.config.trans_num_data.keys():
                    raise ConfigurationError(
                        """ 
                        Missing a valid trans_num_data configuration to build "trans_num_phase_comp" for {} in {}.  
                        Provide this configuration or use another "trans_num_calculation"
                        configuration value to contruct the demanded variable(s))""".format(
                            j, self.name
                        )
                    )
                else:
                    return (
                        b.trans_num_phase_comp[p, j]
                        == self.params.config.trans_num_data[p, j]
                    )
            else:
                if (p, j) in self.params.config.trans_num_data.keys():
                    _log.warning(
                        """
                        The provided trans_num_data of {} will be overritten by the calculated data for {}
                        because "TransportNumberCalculation" is set as "ElectricalMobility".""".format(
                            j, self.name
                        )
                    )
                return b.trans_num_phase_comp[p, j] == abs(
                    b.charge_comp[j]
                ) * b.elec_mobility_phase_comp[p, j] * b.conc_mol_phase_comp[
                    p, j
                ] / sum(
                    abs(b.charge_comp[j])
                    * b.elec_mobility_phase_comp[p, j]
                    * b.conc_mol_phase_comp[p, j]
                    for j in self.params.ion_set
                )

        self.eq_trans_num_phase_comp = Constraint(
            self.params.phase_list,
            self.params.ion_set,
            rule=rule_trans_num_phase_comp,
        )

    def _equiv_conductivity_phase(self):
        self.equiv_conductivity_phase = Var(
            self.params.phase_list,
            initialize=0.5,
            units=pyunits.meter**2 * pyunits.ohm**-1 * pyunits.mol**-1,
            doc="Total equivalent electrical conducitivty of the liquid phase",
        )

        def rule_equiv_conductivity_phase(b, p):
            if (
                self.params.config.equiv_conductivity_calculation
                == EquivalentConductivityCalculation.none
            ):
                if p not in self.params.config.equiv_conductivity_phase_data.keys():
                    raise ConfigurationError(
                        """ 
                        Missing a valid equiv_conductivity_phase_data configuration to build 
                        "equiv_conductivity_phase" and its derived variables for {}. 
                        Provide this configuration or use another "equiv_conductivity_calculation"
                        configuration value to contruct the demanded variable(s))""".format(
                            self.name
                        )
                    )
                else:
                    if len(self.params.ion_set) > 2:
                        _log.warning(
                            """ 
                            Caution should be taken to use a constant solution equivalent conductivity for a multi-electrolyte system.
                            Heterogeneous concentration variation among ions may lead to varying equivalent conductivity and computing
                            the phase equivalent conductivity using the "EquivalentConductivityCalculation.ElectricalMobility" method 
                            is recommended."""
                        )
                    return (
                        b.equiv_conductivity_phase[p]
                        == self.params.config.equiv_conductivity_phase_data[p]
                        * pyunits.meter**2
                        * pyunits.ohm**-1
                        * pyunits.mol**-1
                    )
            else:
                if len(self.params.config.equiv_conductivity_phase_data) != 0:
                    _log.warning(
                        """
                        The provided equiv_conductivity_phase_data will be overritten by the calculated data for {} because
                        "EquivalentConductivityCalculation" is set as "ElectricalMobility".""".format(
                            self.name
                        )
                    )
                return b.equiv_conductivity_phase[p] == sum(
                    Constants.faraday_constant
                    * abs(b.charge_comp[j])
                    * b.elec_mobility_phase_comp[p, j]
                    * b.conc_mol_phase_comp[p, j]
                    for j in self.params.ion_set
                ) / sum(
                    abs(b.charge_comp[j]) * b.conc_mol_phase_comp[p, j]
                    for j in self.params.cation_set
                )

        self.eq_equiv_conductivity_phase = Constraint(
            self.params.phase_list, rule=rule_equiv_conductivity_phase
        )

    def _elec_cond_phase(self):
        self.elec_cond_phase = Var(
            self.params.phase_list,
            initialize=0.1,
            units=pyunits.ohm**-1 * pyunits.meter**-1,
            doc="Electrical conductivity",
        )

        def rule_elec_cond_phase(b, p):
            return b.elec_cond_phase[p] == b.equiv_conductivity_phase[p] * sum(
                abs(b.charge_comp[j]) * b.conc_mol_phase_comp[p, j]
                for j in self.params.cation_set
            )

        self.eq_elec_cond_phase = Constraint(
            self.params.phase_list, rule=rule_elec_cond_phase
        )

    # -----------------------------------------------------------------------------
    # General Methods
    # NOTE: For scaling in the control volume to work properly, these methods must
    # return a pyomo Var or Expression

    def get_material_flow_terms(self, p, j):
        """Create material flow terms for control volume."""
        return self.flow_mol_phase_comp[p, j]

    # TODO: add enthalpy terms later
    # def get_enthalpy_flow_terms(self, p):
    #     """Create enthalpy flow terms."""
    #     return self.enth_flow

    # TODO: make property package compatible with dynamics
    # def get_material_density_terms(self, p, j):
    #     """Create material density terms."""

    # def get_enthalpy_density_terms(self, p):
    #     """Create enthalpy density terms."""

    def default_material_balance_type(self):
        return MaterialBalanceType.componentTotal

    def default_energy_balance_type(self):
        return EnergyBalanceType.none

    def get_material_flow_basis(self):
        return MaterialFlowBasis.molar

    def define_state_vars(self):
        """Define state vars."""
        return {
            "flow_mass_phase_comp": self.flow_mass_phase_comp,
            "temperature": self.temperature,
            "pressure": self.pressure,
        }

    
    # -----------------------------------------------------------------------------
    # Scaling methods
    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

        # setting scaling factors for variables

        # default scaling factors have already been set with
        # idaes.core.property_base.calculate_scaling_factors()
        # for the following variables: pressure,
        # temperature, dens_mass, visc_d_phase, diffus_phase_comp

        # the following variables should have users' input of scaling factors;
        # missing input triggers a warning
        if iscale.get_scaling_factor(self.flow_mass_phase_comp["Liq", "H2O"]) is None:
            sf = iscale.get_scaling_factor(
                self.flow_mass_phase_comp["Liq", "H2O"], default=1, warning=True
            )
            iscale.set_scaling_factor(self.flow_mass_phase_comp["Liq", "H2O"], sf)

        for j in self.params.solute_set:
            if iscale.get_scaling_factor(self.flow_mass_phase_comp["Liq", j]) is None:
                sf = iscale.get_scaling_factor(
                    self.flow_mass_phase_comp["Liq", j], default=1, warning=True
                )
                iscale.set_scaling_factor(self.flow_mass_phase_comp["Liq", j], sf)

        if self.is_property_constructed("flow_equiv_phase_comp"):
            for j in self.flow_equiv_phase_comp.keys():
                if iscale.get_scaling_factor(self.flow_equiv_phase_comp[j]) is None:
                    sf = iscale.get_scaling_factor(self.flow_mol_phase_comp[j])
                    iscale.set_scaling_factor(self.flow_equiv_phase_comp[j], sf)

        # The following variables and parameters have computed scalling factors;
        # Users do not have to input scaling factors but, if they do, their value
        # will override.
        for j, v in self.mw_comp.items():
            if iscale.get_scaling_factor(v) is None:
                iscale.set_scaling_factor(self.mw_comp[j], value(v) ** -1)
        for ind, v in self.diffus_phase_comp.items():
            if iscale.get_scaling_factor(v) is None:
                if ind in self.params.config.diffusivity_data.keys():
                    sf = self.params.config.diffusivity_data[ind] ** -1
                else:
                    sf = 1e10
                iscale.set_scaling_factor(self.diffus_phase_comp[ind], sf)
        if self.is_property_constructed("dens_mass_solvent"):
            if iscale.get_scaling_factor(self.dens_mass_solvent) is None:
                iscale.set_scaling_factor(self.dens_mass_solvent, 1e-3)
        if self.is_property_constructed("dens_mass_phase"):
            for p, v in self.dens_mass_phase.items():
                if iscale.get_scaling_factor(v) is None:
                    iscale.set_scaling_factor(self.dens_mass_phase[p], 1e-3)
        if self.is_property_constructed("visc_d_phase"):
            for p, v in self.visc_d_phase.items():
                if iscale.get_scaling_factor(v) is None:
                    iscale.set_scaling_factor(self.visc_d_phase[p], 1e3)
        if self.is_property_constructed("visc_k_phase"):
            for p, v in self.visc_k_phase.items():
                if iscale.get_scaling_factor(v) is None:
                    iscale.set_scaling_factor(self.visc_k_phase[p], 1e6)
        if self.is_property_constructed("mole_frac_phase_comp"):
            for j in self.params.component_list:
                if (
                    iscale.get_scaling_factor(self.mole_frac_phase_comp["Liq", j])
                    is None
                ):
                    if j == "H2O":
                        iscale.set_scaling_factor(
                            self.mole_frac_phase_comp["Liq", j], 1
                        )
                    else:
                        sf = iscale.get_scaling_factor(
                            self.flow_mass_phase_comp["Liq", j]
                        ) / iscale.get_scaling_factor(
                            self.flow_mass_phase_comp["Liq", "H2O"]
                        )
                        iscale.set_scaling_factor(
                            self.mole_frac_phase_comp["Liq", j], sf
                        )

        # if self.is_property_constructed("flow_mass_phase_comp"):
        #     for j in self.params.component_list:
        #         if (
        #             iscale.get_scaling_factor(self.flow_mass_phase_comp["Liq", j])
        #             is None
        #         ):
        #             sf = iscale.get_scaling_factor(
        #                 self.flow_mol_phase_comp["Liq", j]
        #             ) * iscale.get_scaling_factor(self.mw_comp[j])
        #             iscale.set_scaling_factor(self.flow_mass_phase_comp["Liq", j], sf)

        if self.is_property_constructed("mass_frac_phase_comp"):
            for j in self.params.component_list:
                comp = self.params.get_component(j)
                if (
                    iscale.get_scaling_factor(self.mass_frac_phase_comp["Liq", j])
                    is None
                ):
                    if comp.is_solute():
                        sf = iscale.get_scaling_factor(
                            self.flow_mass_phase_comp["Liq", j]
                        ) / iscale.get_scaling_factor(
                            self.flow_mass_phase_comp["Liq", "H2O"]
                        )
                        iscale.set_scaling_factor(
                            self.mass_frac_phase_comp["Liq", j], sf
                        )
                    else:
                        iscale.set_scaling_factor(
                            self.mass_frac_phase_comp["Liq", j], 1
                        )

        if self.is_property_constructed("conc_mass_phase_comp"):
            for j in self.params.component_list:
                sf_dens = iscale.get_scaling_factor(self.dens_mass_phase["Liq"])
                if (
                    iscale.get_scaling_factor(self.conc_mass_phase_comp["Liq", j])
                    is None
                ):
                    if j == "H2O":
                        # solvents typically have a mass fraction between 0.5-1
                        iscale.set_scaling_factor(
                            self.conc_mass_phase_comp["Liq", j], sf_dens
                        )
                    else:
                        iscale.set_scaling_factor(
                            self.conc_mass_phase_comp["Liq", j],
                            sf_dens
                            * iscale.get_scaling_factor(
                                self.mass_frac_phase_comp["Liq", j]
                            ),
                        )

        if self.is_property_constructed("conc_mol_phase_comp"):
            for j in self.params.component_list:
                if (
                    iscale.get_scaling_factor(self.conc_mol_phase_comp["Liq", j])
                    is None
                ):
                    sf = iscale.get_scaling_factor(
                        self.conc_mass_phase_comp["Liq", j]
                    ) / iscale.get_scaling_factor(self.mw_comp[j])
                    iscale.set_scaling_factor(self.conc_mol_phase_comp["Liq", j], sf)

        if self.is_property_constructed("conc_equiv_phase_comp"):
            for j in self.params.ion_set:
                if (
                    iscale.get_scaling_factor(self.conc_equiv_phase_comp["Liq", j])
                    is None
                ):
                    sf = iscale.get_scaling_factor(self.conc_mol_phase_comp["Liq", j])
                    iscale.set_scaling_factor(self.conc_equiv_phase_comp["Liq", j], sf)

        if self.is_property_constructed("pressure_osm_phase"):
            if iscale.get_scaling_factor(self.pressure_osm_phase) is None:
                sf = (
                    1e-3
                    * sum(
                        iscale.get_scaling_factor(self.conc_mol_phase_comp["Liq", j])
                        ** 2
                        for j in self.params.ion_set | self.params.solute_set
                    )
                    ** 0.5
                )
                iscale.set_scaling_factor(self.pressure_osm_phase, sf)

        # if self.is_property_constructed("elec_mobility_phase_comp"):
        #     for ind, v in self.elec_mobility_phase_comp.items():
        #         if iscale.get_scaling_factor(v) is None:
        #             if (
        #                 self.params.config.elec_mobility_calculation
        #                 == ElectricalMobilityCalculation.EinsteinRelation
        #             ):
        #                 sf = iscale.get_scaling_factor(self.diffus_phase_comp[ind]) / 40
        #             else:
        #                 sf = self.params.config.elec_mobility_data[ind] ** -1
        #             iscale.set_scaling_factor(self.elec_mobility_phase_comp[ind], sf)

        # if self.is_property_constructed("trans_num_phase_comp"):
        #     for ind, v in self.trans_num_phase_comp.items():
        #         if iscale.get_scaling_factor(v) is None:
        #             iscale.set_scaling_factor(self.trans_num_phase_comp[ind], 10)
        # if self.is_property_constructed("equiv_conductivity_phase"):
        #     for ind, v in self.equiv_conductivity_phase.items():
        #         if iscale.get_scaling_factor(v) is None:
        #             if (
        #                 self.params.config.equiv_conductivity_calculation
        #                 == EquivalentConductivityCalculation.ElectricalMobility
        #             ):
        #                 sf = (
        #                     1
        #                     / 96485
        #                     * sum(
        #                         iscale.get_scaling_factor(
        #                             self.elec_mobility_phase_comp["Liq", j]
        #                         )
        #                         ** 2
        #                         * iscale.get_scaling_factor(
        #                             self.conc_mol_phase_comp["Liq", j]
        #                         )
        #                         ** 2
        #                         for j in self.params.ion_set
        #                     )
        #                     ** 0.5
        #                     / sum(
        #                         iscale.get_scaling_factor(
        #                             self.conc_mol_phase_comp["Liq", j]
        #                         )
        #                         ** 2
        #                         for j in self.params.cation_set
        #                     )
        #                     ** 0.5
        #                 )
        #             else:
        #                 sf = self.params.config.equiv_conductivity_phase_data[ind] ** -1
        #             iscale.set_scaling_factor(self.equiv_conductivity_phase[ind], sf)

        # if self.is_property_constructed("elec_cond_phase"):
        #     if iscale.get_scaling_factor(self.elec_cond_phase) is None:
        #         for ind, v in self.elec_cond_phase.items():
        #             sf = (
        #                 iscale.get_scaling_factor(self.equiv_conductivity_phase[ind])
        #                 * sum(
        #                     iscale.get_scaling_factor(
        #                         self.conc_mol_phase_comp["Liq", j]
        #                     )
        #                     ** 2
        #                     for j in self.params.cation_set
        #                 )
        #                 ** 0.5
        #             )
        #             iscale.set_scaling_factor(self.elec_cond_phase[ind], sf)

        if self.is_property_constructed("flow_vol_phase"):
            sf = (
                iscale.get_scaling_factor(
                    self.flow_mass_phase_comp["Liq", "H2O"], default=1
                )
                / iscale.get_scaling_factor(self.dens_mass_phase["Liq"])
            )
            iscale.set_scaling_factor(self.flow_vol_phase, sf)

        if self.is_property_constructed("flow_vol"):
            sf = iscale.get_scaling_factor(self.flow_vol_phase)
            iscale.set_scaling_factor(self.flow_vol, sf)

        if self.is_property_constructed("molality_phase_comp"):
            for j in self.params.ion_set | self.params.solute_set:
                if (
                    iscale.get_scaling_factor(self.molality_phase_comp["Liq", j])
                    is None
                ):
                    sf = (
                        iscale.get_scaling_factor(self.flow_mol_phase_comp["Liq", j])
                        / iscale.get_scaling_factor(
                            self.flow_mol_phase_comp["Liq", "H2O"]
                        )
                        / iscale.get_scaling_factor(self.mw_comp["H2O"])
                    )
                    iscale.set_scaling_factor(self.molality_phase_comp["Liq", j], sf)

        if self.is_property_constructed("act_coeff_phase_comp"):
            for j in self.params.ion_set | self.params.solute_set:
                if (
                    iscale.get_scaling_factor(self.act_coeff_phase_comp["Liq", j])
                    is None
                ):
                    iscale.set_scaling_factor(self.act_coeff_phase_comp["Liq", j], 1)

        if self.is_property_constructed("debye_huckel_constant"):
            if iscale.get_scaling_factor(self.debye_huckel_constant) is None:
                iscale.set_scaling_factor(self.debye_huckel_constant, 10)

        if self.is_property_constructed("ionic_strength_molal"):
            if iscale.get_scaling_factor(self.ionic_strength_molal) is None:
                sf = min(
                    iscale.get_scaling_factor(self.molality_phase_comp["Liq", j])
                    for j in self.params.ion_set | self.params.solute_set
                )
                iscale.set_scaling_factor(self.ionic_strength_molal, sf)

        # transforming constraints
        transform_property_constraints(self)

        if self.is_property_constructed("debye_huckel_constant"):
            iscale.constraint_scaling_transform(self.eq_debye_huckel_constant, 10)

        if self.is_property_constructed("ionic_strength_molal"):
            iscale.constraint_scaling_transform(self.eq_ionic_strength_molal, 1)
