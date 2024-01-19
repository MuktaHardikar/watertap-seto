import pytest
import os

import pandas as pd
from pyomo.environ import (
    SolverFactory,
    ConcreteModel,
    Var,
    value,
    assert_optimal_termination,
    units as pyunits,
)
from pyomo.network import Port

from watertap_contrib.reflo.solar_models.surrogate.flat_plate import FlatPlateSurrogate
from watertap_contrib.reflo.costing import EnergyCosting

from idaes.core.util.testing import initialization_tester
from idaes.core.solvers import get_solver
from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate
from idaes.core.surrogate.surrogate_block import SurrogateBlock
from idaes.core import FlowsheetBlock, UnitModelCostingBlock
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    number_variables,
    number_total_constraints,
    number_unused_variables,
)
from idaes.core.util.scaling import (
    calculate_scaling_factors,
    unscaled_variables_generator,
)


# Get default solver for testing
solver = get_solver()
solver = SolverFactory("ipopt")
dataset_filename = os.path.join(os.path.dirname(__file__), "data/flat_plate_data.pkl")
surrogate_filename = os.path.join(
    os.path.dirname(__file__), "flat_plate_surrogate.json"
)
expected_heat_annual = [
    2.201e9,
    2.530e9,
    2.266e9,
    1.472e9,
    2.611e9,
    3.567e8,
    3.049e8,
    2.200e9,
    6.599e8,
    1.144e9,
]
expected_electricity_annual = [
    4.889e7,
    5.804e7,
    4.969e7,
    3.245e7,
    5.682e7,
    7.944e6,
    6.800e6,
    4.760e7,
    1.530e7,
    2.569e7,
]


def get_data():
    df = pd.read_pickle(dataset_filename)
    df = df.sample(n=90, random_state=1)  # random_state ensures reproducibility
    return {"training": df[:80], "validation": df[80:90]}


class TestFlatPlate:
    @pytest.fixture(scope="class")
    def flat_plate_frame(self):

        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.flatplate = FlatPlateSurrogate()

        return m

    @pytest.mark.unit
    @pytest.mark.skip
    def test_build(self, flat_plate_frame):
        m = flat_plate_frame

        assert len(m.fs.flatplate.config) == 3
        assert not m.fs.flatplate.config.dynamic
        assert not m.fs.flatplate.config.has_holdup
        assert m.fs.flatplate._tech_type == "flat_plate"
        assert isinstance(m.fs.flatplate.surrogate_blk, SurrogateBlock)

        surr_input_str = ["heat_load", "hours_storage", "temperature_hot"]
        surr_output_str = ["heat_annual", "electricity_annual"]

        assert m.fs.flatplate.input_labels == surr_input_str
        assert m.fs.flatplate.surrogate.input_labels() == surr_input_str
        assert m.fs.flatplate.output_labels == surr_output_str
        assert m.fs.flatplate.surrogate.output_labels() == surr_output_str
        assert m.fs.flatplate.surrogate_file.lower() == surrogate_filename.lower()
        assert m.fs.flatplate.dataset_filename.lower() == dataset_filename.lower()
        assert m.fs.flatplate.surrogate.n_inputs() == 3
        assert m.fs.flatplate.surrogate.n_outputs() == 2

        for s in surr_input_str + surr_output_str:
            v = getattr(m.fs.flatplate, s)
            assert isinstance(v, Var)

        no_ports = list()
        for c in m.fs.flatplate.component_objects():
            if isinstance(c, Port):
                no_ports.append(c)
        assert len(no_ports) == 0
        assert number_variables(m.fs.flatplate) == 10
        assert number_unused_variables(m.fs.flatplate) == 0
        assert number_total_constraints(m.fs.flatplate) == 7

    @pytest.mark.unit
    @pytest.mark.skip
    def test_surrogate_variable_bounds(self, flat_plate_frame):
        m = flat_plate_frame
        assert m.fs.flatplate.heat_load.bounds == tuple([100, 1000])
        assert m.fs.flatplate.hours_storage.bounds == tuple([0, 26])
        assert m.fs.flatplate.temperature_hot.bounds == tuple([50, 100])


    @pytest.mark.unit
    @pytest.mark.skip
    def test_dof(self, flat_plate_frame):

        m = flat_plate_frame
        m.fs.flatplate.heat_load.fix(500)
        m.fs.flatplate.hours_storage.fix(12)
        m.fs.flatplate.temperature_hot.fix(70)
        assert degrees_of_freedom(m) == 0

    @pytest.mark.unit
    @pytest.mark.skip
    def test_calculate_scaling(self, flat_plate_frame):

        m = flat_plate_frame
        calculate_scaling_factors(m)
        assert len(list(unscaled_variables_generator(m))) == 0

    @pytest.mark.component
    @pytest.mark.skip
    def test_initialization(self, flat_plate_frame):
        initialization_tester(flat_plate_frame, unit=flat_plate_frame.fs.flatplate)

    @pytest.mark.component
    @pytest.mark.skip
    def test_solve(self, flat_plate_frame):
        results = solver.solve(flat_plate_frame)
        assert_optimal_termination(results)

    @pytest.mark.component
    @pytest.mark.skip
    def test_costing(self, flat_plate_frame):
        m = flat_plate_frame
        m.fs.test_flow = 50 * pyunits.Mgallons / pyunits.day

        m.fs.costing = EnergyCosting()
        m.fs.flatplate.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.costing
        )

        m.fs.costing.factor_maintenance_labor_chemical.fix(0)
        m.fs.costing.factor_total_investment.fix(1)

        m.fs.costing.cost_process()
        m.fs.costing.add_LCOW(flow_rate=m.fs.test_flow)

        results = solver.solve(m)
        assert_optimal_termination(results)
