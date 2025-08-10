from idaes.core.solvers import get_solver
from pyomo.environ import SolverFactory, value
import pandas as pd
from watertap_contrib.reflo.analysis.case_studies.permian.permian_RPT1_MD import (
    run_permian_rpt1_md,
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
    'water_recovery': [0.59,0.478,0.23]
    }   
    
    input_dict = {
        'Qin': 5, 
        'tds': 130,
        'water_recovery':0.5,
        'grid_frac_heat':1,
        'heat_price':0.0166,
        "electricity_price":0.04346,
        'dwi_lcow': 8.4,
        'cst_cost_per_total_aperture_area':297,
        'cst_cost_per_storage_capital': 62,
        'cost_per_land_area':4000
    }


    #############################################################################################
    # Select sweep type
    #############################################################################################
    
    sweep_type = "water_recovery"
    only_plot = False
    # only_plot = True

    xcol_dict = {
        "water_recovery":"fs.water_recovery",
        "heat_price": "fs.costing.heat_cost_buy",
        "hours_storage": "fs.energy.FPC.hours_storage",
        "grid_frac_heat": "fs.costing.frac_heat_from_grid",
        "dwi_lcow":"fs.treatment.costing.deep_well_injection.dwi_lcow",
        'cst_cost_per_total_aperture_area':'fs.energy.costing.trough_surrogate.cost_per_total_aperture_area',
        'cst_cost_per_storage_capital':'fs.energy.costing.trough_surrogate.cost_per_storage_capital',
    }

    ax_dict = {
        "water_recovery": "MD Water Recovery (%)",
        "heat_price": "Heat Price ($/kWh)",
        "hours_storage": "Hours Storage (h)",
        "grid_frac_heat": "Grid Fraction (Heat)",
        "dwi_lcow": "DWI LCOW (\$/m$^3$)",
        'cst_cost_per_total_aperture_area':"Cost per Total Aperture Area ($/m2)",
        'cst_cost_per_storage_capital':"Cost per Thermal Storage Capacity ($/kWh)",
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
 

    m = run_permian_rpt1_md(
            Qin=input_dict['Qin'], 
            tds=input_dict['tds'], 
            grid_frac_heat=input_dict['grid_frac_heat'],
            water_recovery= input_dict['water_recovery'],
            heat_price=input_dict['heat_price'],
            electricity_price=input_dict['electricity_price'],
            dwi_lcow=input_dict['dwi_lcow'],
            cst_cost_per_total_aperture_area=input_dict['cst_cost_per_total_aperture_area'],
            cst_cost_per_storage_capital=input_dict['cst_cost_per_storage_capital'],
            cost_per_land_area = input_dict['cost_per_land_area'],
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


            m = run_permian_rpt1_md(
                Qin=input_dict['Qin'], 
                tds=input_dict['tds'], 
                grid_frac_heat=input_dict['grid_frac_heat'],
                water_recovery= input_dict['water_recovery'],
                heat_price=input_dict['heat_price'],
                electricity_price=input_dict['electricity_price'],
                dwi_lcow=input_dict['dwi_lcow'],
                cst_cost_per_total_aperture_area=input_dict['cst_cost_per_total_aperture_area'],
                cst_cost_per_storage_capital=input_dict['cst_cost_per_storage_capital'],
                cost_per_land_area = input_dict['cost_per_land_area'],
                )
        
            results_dict_test = results_dict_append(m, results_dict_test)

    try:
        df = pd.DataFrame.from_dict(results_dict_test)
    except ValueError:
        df = pd.DataFrame.from_dict(results_dict_test, orient='index')
    
    
    filename = "/Users/mhardika/Documents/watertap-seto/Mukta-Work/permian-case-study-md/ST1_MD_sweep_results//permian_RPT1_MD_flow_tds_sweep.csv"
    df.to_csv(filename)


    