
# General python imports
import numpy as np
import pandas as pd
import logging
from collections import deque

# Pyomo imports
from pyomo.environ import Set, Expression, value, Objective

# IDAES imports
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
from idaes.core.solvers.get_solver import get_solver
import idaes.logger as idaeslog
# Flowsheet function imports
from PV_RO_battery_flowsheet import (
    build_pv_battery_flowsheet,
    fix_dof_and_initialize,
)
import matplotlib.pyplot as plt

__author__ = "Zhuoran Zhang"

_log = idaeslog.getLogger(__name__)
solver = get_solver()

def get_pv_ro_variable_pairs(t1, t2):
    """
    This function returns pairs of variables that need to be connected across two time periods

    Args:
        t1: current time block
        t2: next time block

    Returns:
        None
    """
    return [
        (t1.fs.battery.state_of_charge[0], t2.fs.battery.initial_state_of_charge),
        (t1.fs.battery.energy_throughput[0], t2.fs.battery.initial_energy_throughput),
        (t1.fs.battery.nameplate_power, t2.fs.battery.nameplate_power),
        (t1.fs.battery.nameplate_energy, t2.fs.battery.nameplate_energy),
        (t1.fs.pv.size, t2.fs.pv.size)]

def unfix_dof(m):
    """
    This function unfixes a few degrees of freedom for optimization

    Args:
        m: object containing the integrated nuclear plant flowsheet

    Returns:
        None
    """
    m.fs.battery.nameplate_energy.unfix()
    m.fs.battery.nameplate_power.unfix()
    # m.fs.battery.initial_state_of_charge.unfix()
    # m.fs.battery.initial_energy_throughput.unfix()
    return

# PV surrogate output for 4 select days
file_path = '/Users/mhardika/Documents/watertap_seto/watertap-seto/src/watertap_contrib/seto/analysis/multiperiod'

summer_pv_df = pd.read_csv(file_path +'/data_files/summer_pv_2000kw.csv',index_col='time (h)')
spring_pv_df = pd.read_csv(file_path +'/data_files/spring_pv_2000kw.csv',index_col='time (h)')
fall_pv_df = pd.read_csv(file_path +'/data_files/fall_pv_2000kw.csv',index_col='time (h)')
winter_pv_df = pd.read_csv(file_path +'/data_files/winter_pv_2000kw.csv',index_col='time (h)')

# Arbitrary electricity costs
elec_price_df = pd.read_csv(file_path +'/data_files/elec_price.csv',index_col='time (h)')
elec_price = np.array(elec_price_df['elec_price'].values)
elec_price = np.append(elec_price,elec_price)
elec_price = np.append(elec_price,elec_price)

pv_gen = summer_pv_df['power (kW)'].values,
pv_gen = np.append(pv_gen,fall_pv_df['power (kW)'].values)
pv_gen = np.append(pv_gen,winter_pv_df['power (kW)'].values)
pv_gen = np.append(pv_gen,spring_pv_df['power (kW)'].values)

def create_multiperiod_pv_battery_model(
        n_time_points= 96,
        ro_capacity = 6000, # m3/day
        ro_elec_req = 944.3, # kW
        cost_battery_power = 75, # $/kW
        cost_battery_energy = 50, # $/kWh      
        pv_gen = pv_gen,
        # Zhuoran was using GHI to calculate pv_gen
        # 24-hr GHI in Phoenix, AZ on June 18th (W/m2)
        # GHI = [0, 0, 0, 0, 0, 23, 170, 386, 596, 784, 939, 1031, 1062, 1031, 938, 790, 599, 383, 166, 31, 0, 0, 0, 0],
        # elec_price = [0.07] * 24,
        elec_price = elec_price
    ):
    
    """
    This function creates a multi-period pv battery flowsheet object. This object contains 
    a pyomo model with a block for each time instance.

    Args:
        n_time_points: Number of time blocks to create

    Returns:
        Object containing multi-period vagmd batch flowsheet model
    """
    mp = MultiPeriodModel(
        n_time_points=n_time_points,
        process_model_func=build_pv_battery_flowsheet,
        linking_variable_func=get_pv_ro_variable_pairs,
        initialization_func=fix_dof_and_initialize,
        unfix_dof_func=unfix_dof,
        outlvl=logging.WARNING,
    )

    flowsheet_options={ t: { 
                            # "GHI" : GHI[t],
                            "pv_gen": pv_gen[t], 
                            "elec_price": elec_price[t],
                            "ro_capacity": ro_capacity, 
                            "ro_elec_req": ro_elec_req} 
                            for t in range(n_time_points)
    }

    # create the multiperiod object
    mp.build_multi_period_model(
        model_data_kwargs=flowsheet_options,
        flowsheet_options={ "ro_capacity": ro_capacity, 
                            "ro_elec_req": ro_elec_req},
        initialization_options=None,
        unfix_dof_options=None,
        )

    # initialize the beginning status of the system
    mp.blocks[0].process.fs.battery.initial_state_of_charge.fix(0)
    mp.blocks[23].process.fs.battery.initial_state_of_charge.fix(10)
    mp.blocks[47].process.fs.battery.initial_state_of_charge.fix(10)
    mp.blocks[71].process.fs.battery.initial_state_of_charge.fix(10)
    mp.blocks[95].process.fs.battery.initial_state_of_charge.fix(10)
    mp.blocks[0].process.fs.battery.initial_energy_throughput.fix(0)

    # Add battery cost function
    @mp.Expression(doc="battery cost")
    def battery_cost(b):
        return ( 0.096 * # capital recovery factor
            (cost_battery_power * b.blocks[0].process.fs.battery.nameplate_power
            +cost_battery_energy * b.blocks[0].process.fs.battery.nameplate_energy))
        
    # Add PV cost function
    @mp.Expression(doc="PV cost")
    def pv_cost(b):
        return (
            1040 * b.blocks[0].process.fs.pv.size * 0.096 # Annualized CAPEX
            + 9 * b.blocks[0].process.fs.pv.size)          # OPEX

    # Total cost
    @mp.Expression(doc='total cost')
    def total_cost(b):
        # The annualized capital cost is evenly distributed to the multiperiod
        return (
            (b.battery_cost + b.pv_cost) / 365 / 24 * n_time_points
            + sum([b.blocks[i].process.grid_cost for i in range(n_time_points)])
        )

    # LCOW
    @mp.Expression(doc='total cost')
    def LCOW(b):
        # LCOW from RO: 0.45
        return (
            b.total_cost / ro_capacity / 24 * n_time_points + 0.45
        )   

    # Set objective
    mp.obj = Objective(expr=mp.LCOW)

    return mp

if __name__ == "__main__":
    mp = create_multiperiod_pv_battery_model()
    results = solver.solve(mp)
    n = 96
    # for i in range(n):
    #     print(f'battery status at hour: {i}', value(mp.blocks[i].process.fs.battery.state_of_charge[0]))    
    #     print('pv gen(kW): ', value(mp.blocks[i].process.fs.curtailment))
    print('pv size: ', value(mp.blocks[0].process.fs.pv.size))
    print('battery power: ', value(mp.blocks[0].process.fs.battery.nameplate_power))
    print('battery energy: ', value(mp.blocks[0].process.fs.battery.nameplate_energy))
    print('total cost: ', value(mp.LCOW))

    # Create diagrams
    # plt.clf()
    fig,  axes= plt.subplots(2, figsize=(8,6))
    (ax1, ax2) = axes
    hour = [i for i in range(96)]
    battery_state = [value(mp.blocks[i].process.fs.battery.state_of_charge[0]) for i in range(n)]
    # pv_gen = [value(mp.blocks[i].process.fs.pv.elec_generation) for i in range(48)]
    pv_curtail = [value(mp.blocks[i].process.fs.curtailment) for i in range(n)]

    ax1.plot(hour, battery_state, 'r', label='Battery state (kWh)')
    ax1.plot(hour, pv_gen, 'k', label = 'PV generation (kWh)')
    ax1.plot(hour, pv_curtail, 'g', label = 'PV curtailment (kWh)')
    ax1.vlines(x=[23,47,71],ymin=0,ymax=6000,linestyle='--',color='black')
    ax1.set_ylim([0,6000])

    ax3 = ax1.twinx()
    ax3.plot(hour, elec_price,'--',label='Grid Price')
    ax3.set_ylabel('Grid Price ($/kWh)')
    ax3.set_ylim([0,3])
    ax3.legend(loc="upper right", frameon = False, fontsize = 'small')

    # ax1.set_xlabel('Hour (June 18th)')
    ax1.set_ylabel('Energy (kWh)')
    ax1.legend(loc="upper left", frameon = False, fontsize = 'small')

    pv_to_ro = [value(mp.blocks[i].process.fs.pv_to_ro) for i in range(n)]
    battery_to_ro = [value(mp.blocks[i].process.fs.battery.elec_out[0]) for i in range(n)]
    grid_to_ro = [value(mp.blocks[i].process.fs.grid_to_ro) for i in range(n)]
    labels=["PV to RO", "Battery to RO", "Grid to RO"]
    # ax2.set_xlabel('Hour (June 18th)')
    ax2.set_ylabel('Energy (kWh)')
    ax2.stackplot(hour, pv_to_ro, battery_to_ro, grid_to_ro, labels=labels)
    ax2.legend(loc="upper left", frameon = False, fontsize = 'small')
    ax2.vlines(x=[23,47,71],ymin=0,ymax=1000,linestyle='--',color='black')
    ax2.set_ylim([0,1000])

    plt.show()