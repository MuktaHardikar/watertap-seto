from idaes.core.solvers import get_solver
from pyomo.environ import SolverFactory, value
import pandas as pd
from watertap_contrib.reflo.analysis.case_studies.permian.permian_ZLD1_MD import (
    run_permian_zld1_md,
)
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as mpatches
from watertap_contrib.reflo.analysis.case_studies.permian.utils.results_dict import *
from watertap_contrib.reflo.analysis.case_studies.permian.utils.permian_MD_sweep_plot_functions.case_study_plotting import *



if __name__ == "__main__":

    sweep_dict = {
    'Qin':[1,5,9],
    'tds': [100,130,200],
    'water_recovery': [0.59,0.475,0.23]
    }
    
    input_dict = {
        'Qin': 5, 
        'tds': 130,
        'water_recovery':0.5,
        'grid_frac_heat':1,
        'heat_price':0.0166,
        "electricity_price":0.04346,
        'cst_cost_per_total_aperture_area':297,
        'cst_cost_per_storage_capital': 62,
        'cost_per_land_area':4000,
        'nacl_recovery_price':0,
    }

    permian_cryst_config = {
        "operating_pressures": [0.45, 0.25, 0.208, 0.095], # Operating pressure of each effect (bar)
        "nacl_yield": 0.9, # Yield
        "heat_transfer_coefficient": 1300
        }


    #############################################################################################
    # Select sweep type
    #############################################################################################
    

    xcol_dict = {
        "water_recovery":"fs.water_recovery",
        "heat_price": "fs.costing.heat_cost_buy",
        "hours_storage": "fs.energy.FPC.hours_storage",
        "grid_frac_heat": "fs.costing.frac_heat_from_grid",
        'cst_cost_per_total_aperture_area':'fs.energy.costing.trough_surrogate.cost_per_total_aperture_area',
        'cst_cost_per_storage_capital':'fs.energy.costing.trough_surrogate.cost_per_storage_capital',
        "nacl_recovery_price":"fs.treatment.costing.nacl_recovered.cost",
    }

    ax_dict = {
        "water_recovery": "MD Water Recovery (%)",
        "heat_price": "Heat Price ($/kWh)",
        "hours_storage": "Hours Storage (h)",
        "grid_frac_heat": "Grid Fraction (Heat)",
        'cst_cost_per_total_aperture_area':"Cost per Total Aperture Area ($/m2)",
        'cst_cost_per_storage_capital':"Cost per Thermal Storage Capacity ($/kWh)",
        "nacl_recovery_price": "NaCl Recovery Price ($/kg)",
    }


    skips = [
        "bpe_",
        "dh_vap_w_param",
        "cp_phase_param",
        "pressure_sat_param",
        "enth_mass_param",
        "osm_coeff_param",
        "diffus_param",
        "visc_d_param",
        "diffus_phase",
        "dens_mass_param",
        "therm_cond_phase_param",
        "TIC",
        "TPEC",
        "blocks[",
        "yearly_heat_production",
        "yearly_electricity_production",
        "cp_vap_param",
        "cp_mass_phase",
        "LCOW_component_direct_capex",
        "LCOW_component_indirect_capex",
        "LCOW_component_fixed_opex",
        "LCOW_component_variable_opex"
    ]
 


    m = run_permian_zld1_md(
            Qin=input_dict['Qin'], 
            tds=input_dict['tds'], 
            grid_frac_heat=input_dict['grid_frac_heat'],
            water_recovery= input_dict['water_recovery'],
            heat_price=input_dict['heat_price'],
            electricity_price=input_dict['electricity_price'],
            permian_cryst_config=permian_cryst_config,
            cost_per_total_aperture_area=input_dict['cst_cost_per_total_aperture_area'],
            cost_per_storage_capital=input_dict['cst_cost_per_storage_capital'],
            cost_per_land_area = input_dict['cost_per_land_area'],
            nacl_recovery_price = input_dict['nacl_recovery_price'],
            )

    results_dict_test = build_results_dict(m, skips=skips)

        
    count = 0     
    for i in range(len(sweep_dict["tds"])):
        for j in sweep_dict["Qin"]:
            input_dict["tds"] = sweep_dict["tds"][i]
            input_dict["water_recovery"] = sweep_dict['water_recovery'][i]
            input_dict["Qin"] = j
            count += 1

            # print scenarions
            print("Scenario {}".format(count), input_dict["tds"], input_dict["water_recovery"], input_dict["Qin"])
            try:
                m = run_permian_zld1_md(
                    Qin=input_dict['Qin'], 
                    tds=input_dict['tds'], 
                    grid_frac_heat=input_dict['grid_frac_heat'],
                    water_recovery= input_dict['water_recovery'],
                    heat_price=input_dict['heat_price'],
                    electricity_price=input_dict['electricity_price'],
                    permian_cryst_config=permian_cryst_config,
                    cost_per_total_aperture_area=input_dict['cst_cost_per_total_aperture_area'],
                    cost_per_storage_capital=input_dict['cst_cost_per_storage_capital'],
                    cost_per_land_area = input_dict['cost_per_land_area'],
                    nacl_recovery_price = input_dict['nacl_recovery_price'],
                    )
            
                results_dict_test = results_dict_append(m, results_dict_test)
            except:
                pass

    try:
        df = pd.DataFrame.from_dict(results_dict_test)
    except ValueError:
        df = pd.DataFrame.from_dict(results_dict_test, orient='index')


    filename = "/Users/mhardika/Documents/watertap-seto/Mukta-Work/permian-case-study-md/ST2_MD_sweep_results//permian_ZLD1_MD_flow_tds_sweep.csv"
    df.to_csv(filename)
    

    