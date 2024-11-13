#################################################################################
# WaterTAP Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################

from pyomo.common.config import ConfigValue
import pyomo.environ as pyo

from idaes.core import declare_process_block_class

from watertap.costing.watertap_costing_package import (
    WaterTAPCostingData,
    WaterTAPCostingBlockData,
)
from watertap.costing.zero_order_costing import _load_case_study_definition

from watertap_contrib.reflo.core import PySAMWaterTAP
from watertap_contrib.reflo.solar_models.surrogate.pv.pv_surrogate import (
    PVSurrogateData,
)
from watertap_contrib.reflo.costing.tests.costing_dummy_units import (
    DummyElectricityUnitData,
)


@declare_process_block_class("REFLOCosting")
class REFLOCostingData(WaterTAPCostingData):

    CONFIG = WaterTAPCostingData.CONFIG()
    CONFIG.declare(
        "case_study_definition",
        ConfigValue(
            default=None,
            doc="Path to YAML file defining global parameters for case study. If "
            "not provided, WaterTAP-REFLO values are used.",
        ),
    )

    def build_global_params(self):

        super().build_global_params()

        # Override WaterTAP default value of USD_2018
        self.base_currency = pyo.units.USD_2021

        self.sales_tax_frac = pyo.Param(
            initialize=0.05,
            mutable=True,
            doc="Sales tax as fraction of capital costs",
            units=pyo.units.dimensionless,
        )

        self.heat_cost = pyo.Var(
            initialize=0.0,
            doc="Heat cost",
            units=pyo.units.USD_2018 / pyo.units.kWh,
        )

        self.defined_flows["heat"] = self.heat_cost

        self.heat_cost.fix(0.0)
        self.electricity_cost.fix(0.0)
        self.plant_lifetime.fix(20)
        self.utilization_factor.fix(1)

        # This should override default values
        if self.config.case_study_definition is not None:
            self.case_study_def = _load_case_study_definition(self)
            # Register currency and conversion rates
            if "currency_definitions" in self.case_study_def:
                pyo.units.load_definitions_from_strings(
                    self.case_study_def["currency_definitions"]
                )
            # If currency definition is defined in case study yaml,
            # we should be able to set it here.
            if "base_currency" in self.case_study_def:
                self.base_currency = getattr(
                    pyo.units, self.case_study_def["base_currency"]
                )
            if "base_period" in self.case_study_def:
                self.base_period = getattr(
                    pyo.units, self.case_study_def["base_period"]
                )
            # Define expected flows
            for f, v in self.case_study_def["defined_flows"].items():
                value = v["value"]
                units = getattr(pyo.units, v["units"])
                if self.component(f + "_cost") is not None:
                    self.component(f + "_cost").fix(value * units)
                else:
                    self.defined_flows[f] = value * units


@declare_process_block_class("TreatmentCosting")
class TreatmentCostingData(REFLOCostingData):
    def build_global_params(self):
        super().build_global_params()

    def build_process_costs(self):
        super().build_process_costs()


@declare_process_block_class("EnergyCosting")
class EnergyCostingData(REFLOCostingData):
    def build_global_params(self):
        # If creating an energy unit that generates electricity,
        # set this flag to True in costing package.
        # See PV costing package for example.
        self.has_electricity_generation = False
        super().build_global_params()

    def build_process_costs(self):
        super().build_process_costs()


@declare_process_block_class("REFLOSystemCosting")
class REFLOSystemCostingData(WaterTAPCostingBlockData):

    def build_global_params(self):
        super().build_global_params()

        self.base_currency = pyo.units.USD_2021

        # Fix the parameters
        self.electricity_cost.fix(0.0)
        self.plant_lifetime.fix(20)
        self.utilization_factor.fix(1)

        self.electricity_cost_buy = pyo.Param(
            mutable=True,
            initialize=0.07,
            doc="Electricity cost to buy",
            units=pyo.units.USD_2018 / pyo.units.kWh,
        )

        self.electricity_cost_sell = pyo.Param(
            mutable=True,
            initialize=0.05,
            doc="Electricity cost to sell",
            units=pyo.units.USD_2018 / pyo.units.kWh,
        )

        self.heat_cost_buy = pyo.Param(
            mutable=True,
            initialize=0.07,
            doc="Heat cost to buy",
            units=pyo.units.USD_2018 / pyo.units.kWh,
        )

        self.heat_cost_sell = pyo.Param(
            mutable=True,
            initialize=0.05,
            doc="Heat cost to sell",
            units=pyo.units.USD_2018 / pyo.units.kWh,
        )

        # Build the integrated system costs
        self.build_integrated_costs()

    def build_integrated_costs(self):

        treat_cost = self._get_treatment_cost_block()
        energy_cost = self._get_energy_cost_block()

        # Check if all parameters are equivalent
        self._check_common_param_equivalence(treat_cost, energy_cost)

        # Add all treatment and energy units to _registered_unit_costing
        # so aggregated costs can be calculated at system level.
        for b in [treat_cost, energy_cost]:
            for u in b._registered_unit_costing:
                self._registered_unit_costing.append(u)

        self.total_capital_cost = pyo.Var(
            initialize=1e3,
            domain=pyo.NonNegativeReals,
            doc="Total capital cost for integrated system",
            units=self.base_currency,
        )

        self.total_operating_cost = pyo.Var(
            initialize=1e3,
            doc="Total operating cost for integrated system",
            units=self.base_currency / self.base_period,
        )

        self.aggregate_flow_electricity = pyo.Var(
            initialize=1e3,
            doc="Aggregated electricity flow",
            units=pyo.units.kW,
        )

        self.total_electric_operating_cost = pyo.Var(
            initialize=1e3,
            doc="Total electricity related operating cost",
            units=self.base_currency / self.base_period,
        )

        self.total_heat_operating_cost = pyo.Var(
            initialize=1e3,
            doc="Total heat related operating cost",
            units=self.base_currency / self.base_period,
        )

        if all(hasattr(b, "aggregate_flow_heat") for b in [treat_cost, energy_cost]):
            self.aggregate_flow_heat = pyo.Var(
                initialize=1e3,
                doc="Aggregated heat flow",
                units=pyo.units.kW,
            )

        self.total_capital_cost_constraint = pyo.Constraint(
            expr=self.total_capital_cost
            == pyo.units.convert(
                treat_cost.total_capital_cost + energy_cost.total_capital_cost,
                to_units=self.base_currency,
            )
        )

        self.total_operating_cost_constraint = pyo.Constraint(
            expr=self.total_operating_cost
            == pyo.units.convert(
                treat_cost.total_operating_cost
                + energy_cost.total_operating_cost
                + self.total_electric_operating_cost
                + self.total_heat_operating_cost,
                to_units=self.base_currency / self.base_period,
            )
        )

        if all(hasattr(b, "aggregate_flow_heat") for b in [treat_cost, energy_cost]):
            self.frac_heat_from_grid = pyo.Var(
                initialize=0,
                domain=pyo.NonNegativeReals,
                bounds=(0, 1.00001),
                doc="Fraction of heat from grid",
                units=pyo.units.dimensionless,
            )

        self.frac_elec_from_grid = pyo.Var(
            initialize=0,
            domain=pyo.NonNegativeReals,
            bounds=(0, 1.00001),
            doc="Fraction of electricity from grid",
            units=pyo.units.dimensionless,
        )

        self.aggregate_flow_electricity_purchased = pyo.Var(
            initialize=0,
            domain=pyo.NonNegativeReals,
            doc="Aggregated electricity consumed",
            units=pyo.units.kW,
        )

        self.aggregate_flow_electricity_sold = pyo.Var(
            initialize=0,
            domain=pyo.NonNegativeReals,
            doc="Aggregated electricity produced",
            units=pyo.units.kW,
        )

        self.aggregate_flow_heat_purchased = pyo.Var(
            initialize=0,
            domain=pyo.NonNegativeReals,
            doc="Aggregated heat consumed",
            units=pyo.units.kW,
        )

        self.aggregate_flow_heat_sold = pyo.Var(
            initialize=0,
            domain=pyo.NonNegativeReals,
            doc="Aggregated heat produced",
            units=pyo.units.kW,
        )

        # energy producer's electricity flow is negative
        self.aggregate_electricity_balance = pyo.Constraint(
            expr=(
                self.aggregate_flow_electricity_purchased
                + -1 * energy_cost.aggregate_flow_electricity
                == treat_cost.aggregate_flow_electricity
                + self.aggregate_flow_electricity_sold
            )
        )

        # Calculate fraction of electricity from grid when an electricity generating unit is present
        if energy_cost.has_electricity_generation:
            elec_gen_unit = self._get_electricity_generation_unit()
            self.frac_elec_from_grid_constraint = pyo.Constraint(
                expr=(
                    self.frac_elec_from_grid
                    == 1
                    - (
                        elec_gen_unit.electricity
                        / (
                            elec_gen_unit.electricity
                            + self.aggregate_flow_electricity_purchased
                        )
                    )
                )
            )
        else:
            self.frac_elec_from_grid.fix(1)

        self.aggregate_electricity_complement = pyo.Constraint(
            expr=self.aggregate_flow_electricity_purchased
            * self.aggregate_flow_electricity_sold
            == 0
        )

        if all(hasattr(b, "aggregate_flow_heat") for b in [treat_cost, energy_cost]):

            # treatment block is consuming heat and energy block is generating it

            self.aggregate_flow_heat_constraint = pyo.Constraint(
                expr=self.aggregate_flow_heat
                == self.aggregate_flow_heat_purchased - self.aggregate_flow_heat_sold
            )

            # energy producer's heat flow is negative
            self.aggregate_heat_balance = pyo.Constraint(
                expr=(
                    self.aggregate_flow_heat_purchased
                    + -1 * energy_cost.aggregate_flow_heat
                    == treat_cost.aggregate_flow_heat + self.aggregate_flow_heat_sold
                )
            )

            self.frac_heat_from_grid_constraint = pyo.Constraint(
                expr=(
                    self.frac_heat_from_grid
                    == 1
                    - (
                        -1
                        * energy_cost.aggregate_flow_heat
                        / treat_cost.aggregate_flow_heat
                    )
                )
            )

            self.aggregate_heat_complement = pyo.Constraint(
                expr=self.aggregate_flow_heat_purchased * self.aggregate_flow_heat_sold
                == 0
            )

        elif hasattr(treat_cost, "aggregate_flow_heat"):

            # treatment block is consuming heat but energy block isn't generating
            # we still want to cost the heat consumption

            self.aggregate_flow_heat_sold.fix(0)

            self.aggregate_heat_balance = pyo.Constraint(
                expr=(
                    self.aggregate_flow_heat_purchased == treat_cost.aggregate_flow_heat
                )
            )

        else:
            # treatment block isn't consuming heat and energy block isn't generating
            self.aggregate_flow_heat_purchased.fix(0)
            self.aggregate_flow_heat_sold.fix(0)

        # positive is for cost and negative for revenue
        self.total_electric_operating_cost_constraint = pyo.Constraint(
            expr=self.total_electric_operating_cost
            == (
                pyo.units.convert(
                    self.aggregate_flow_electricity_purchased,
                    to_units=pyo.units.kWh / pyo.units.year,
                )
                * self.electricity_cost_buy
                - pyo.units.convert(
                    self.aggregate_flow_electricity_sold,
                    to_units=pyo.units.kWh / pyo.units.year,
                )
                * self.electricity_cost_sell
            )
        )

        # positive is for cost and negative for revenue
        self.total_heat_operating_cost_constraint = pyo.Constraint(
            expr=self.total_heat_operating_cost
            == (
                pyo.units.convert(
                    self.aggregate_flow_heat_purchased,
                    to_units=pyo.units.kWh / pyo.units.year,
                )
                * self.heat_cost_buy
                - pyo.units.convert(
                    self.aggregate_flow_heat_sold,
                    to_units=pyo.units.kWh / pyo.units.year,
                )
                * self.heat_cost_sell
            )
        )

        # positive is for consumption
        self.aggregate_flow_electricity_constraint = pyo.Constraint(
            expr=self.aggregate_flow_electricity
            == self.aggregate_flow_electricity_purchased
            - self.aggregate_flow_electricity_sold
        )

    def build_process_costs(self):
        """
        Not used in place of build_integrated_costs
        """
        pass

    def add_LCOW(self, flow_rate, name="LCOW"):
        """
        Add Levelized Cost of Water (LCOW) to costing block.
        Args:
            flow_rate - flow rate of water (volumetric) to be used in
                        calculating LCOW
            name (optional) - name for the LCOW variable (default: LCOW)
        """

        LCOW = pyo.Var(
            doc=f"Levelized Cost of Water based on flow {flow_rate.name}",
            units=self.base_currency / pyo.units.m**3,
        )
        self.add_component(name, LCOW)

        LCOW_constraint = pyo.Constraint(
            expr=LCOW
            == (
                self.total_capital_cost * self.capital_recovery_factor
                + self.total_operating_cost
            )
            / (
                pyo.units.convert(flow_rate, to_units=pyo.units.m**3 / self.base_period)
                * self.utilization_factor
            ),
            doc=f"Constraint for Levelized Cost of Water based on flow {flow_rate.name}",
        )
        self.add_component(name + "_constraint", LCOW_constraint)

    def add_LCOE(self, e_model="pysam"):
        """
        Add Levelized Cost of Energy (LCOE) to costing block.
        Args:
            e_model - energy modeling approach used (PySAM or surrogate)
        """

        if e_model == "pysam":
            pysam = self._get_pysam()

            if not pysam._has_been_run:
                raise RuntimeError(
                    f"PySAM model {pysam._pysam_model_name} has not yet been run, so there is no annual_energy data available."
                    "You must run the PySAM model before adding LCOE metric."
                )

            energy_cost = self._get_energy_cost_block()

            self.annual_energy_generated = pyo.Param(
                initialize=pysam.annual_energy,
                units=pyo.units.kWh / pyo.units.year,
                doc=f"Annual energy generated by {pysam._pysam_model_name}",
            )
            LCOE_expr = pyo.Expression(
                expr=(
                    energy_cost.total_capital_cost * self.capital_recovery_factor
                    + (
                        energy_cost.aggregate_fixed_operating_cost
                        + energy_cost.aggregate_variable_operating_cost
                    )
                )
                / self.annual_energy_generated
                * self.utilization_factor
            )
            self.add_component("LCOE", LCOE_expr)

        else:
            raise NotImplementedError(
                "add_LCOE for surrogate models not available yet."
            )

    def add_specific_electric_energy_consumption(self, flow_rate):
        """
        Add specific electric energy consumption (kWh/m**3) to costing block.
        Args:
            flow_rate - flow rate of water (volumetric) to be used in
                        calculating specific energy consumption
        """

        specific_electric_energy_consumption = pyo.Var(
            initialize=100,
            doc=f"Specific electric energy consumption based on flow {flow_rate.name}",
        )

        self.add_component(
            "specific_electric_energy_consumption", specific_electric_energy_consumption
        )

        specific_electric_energy_consumption_constraint = pyo.Constraint(
            expr=specific_electric_energy_consumption
            == self.aggregate_flow_electricity
            / pyo.units.convert(flow_rate, to_units=pyo.units.m**3 / pyo.units.hr)
        )

        self.add_component(
            "specific_electric_energy_consumption_constraint",
            specific_electric_energy_consumption_constraint,
        )

    def add_specific_thermal_energy_consumption(self, flow_rate):
        """
        Add specific thermal energy consumption (kWh/m**3) to costing block.
        Args:
            flow_rate - flow rate of water (volumetric) to be used in
                        calculating specific energy consumption
        """

        specific_thermal_energy_consumption = pyo.Var(
            initialize=100,
            doc=f"Specific thermal energy consumption based on flow {flow_rate.name}",
        )

        self.add_component(
            "specific_thermal_energy_consumption", specific_thermal_energy_consumption
        )

        specific_thermal_energy_consumption_constraint = pyo.Constraint(
            expr=specific_thermal_energy_consumption
            == self.aggregate_flow_heat
            / pyo.units.convert(flow_rate, to_units=pyo.units.m**3 / pyo.units.hr)
        )

        self.add_component(
            "specific_thermal_energy_consumption_constraint",
            specific_thermal_energy_consumption_constraint,
        )

    def _check_common_param_equivalence(self, treat_cost, energy_cost):
        """
        Check if the common costing parameters across all three costing packages
        (treatment, energy, and system) have the same value.
        """

        common_params = [
            "electricity_cost",
            "heat_cost",
            "electrical_carbon_intensity",
            "maintenance_labor_chemical_factor",
            "plant_lifetime",
            "utilization_factor",
            "base_currency",
            "base_period",
            "sales_tax_frac",
            "TIC",
            "TPEC",
        ]

        for cp in common_params:
            tp = getattr(treat_cost, cp)
            ep = getattr(energy_cost, cp)
            if not pyo.value(tp) == pyo.value(ep):
                err_msg = f"The common costing parameter {cp} was found to have a different value "
                err_msg += f"on the energy ({pyo.value(ep)}) and treatment ({pyo.value(tp)}) costing blocks. "
                err_msg += "Common costing parameters must be equivalent across all costing blocks "
                err_msg += "to use REFLOSystemCosting."
                raise ValueError(err_msg)
            if hasattr(self, cp):
                # if REFLOSystemCosting has this parameter,
                # we fix it to the treatment costing block value
                p = getattr(self, cp)
                if isinstance(p, pyo.Var):
                    p.fix(pyo.value(tp))
                elif isinstance(p, pyo.Param):
                    p.set_value(pyo.value(tp))

    def _get_treatment_cost_block(self):
        tb = None
        for b in self.model().component_objects(pyo.Block):
            if isinstance(b, TreatmentCostingData):
                tb = b
        if tb is None:
            err_msg = "REFLOSystemCosting package requires a TreatmentCosting block"
            err_msg += " but one was not found."
            raise ValueError(err_msg)
        else:
            return tb

    def _get_energy_cost_block(self):
        eb = None
        for b in self.model().component_objects(pyo.Block):
            if isinstance(b, EnergyCostingData):
                eb = b
        if eb is None:
            err_msg = "REFLOSystemCosting package requires a EnergyCosting block"
            err_msg += " but one was not found."
            raise ValueError(err_msg)
        else:
            return eb

    def _get_electricity_generation_unit(self):
        elec_gen_unit = None
        for b in self.model().component_objects(pyo.Block):
            if isinstance(
                b, PVSurrogateData
            ):  # PV is only electricity generation model currently
                elec_gen_unit = b
            if isinstance(b, DummyElectricityUnitData):  # only used for testing
                elec_gen_unit = b
        if elec_gen_unit is None:
            err_msg = (
                f"{self.name} indicated an electricity generation model was present "
            )
            err_msg += "on the flowsheet, but none was found."
            raise ValueError(err_msg)
        else:
            return elec_gen_unit

    def _get_pysam(self):
        pysam_block_test_lst = []
        for k, v in vars(self.model()).items():
            if isinstance(v, PySAMWaterTAP):
                pysam_block_test_lst.append(k)

        if len(pysam_block_test_lst) != 1:
            raise Exception("There is no instance of PySAMWaterTAP on this model.")

        else:
            pysam = getattr(self.model(), pysam_block_test_lst[0])
            return pysam
