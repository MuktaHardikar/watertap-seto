from idaes.core.solvers import get_solver
from pyomo.environ import SolverFactory, value
import pandas as pd
from watertap_contrib.reflo.analysis.case_studies.KBHDP.KBHDP_ZLD_MH import (
    zld_main,
)
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as mpatches
from watertap_contrib.reflo.analysis.case_studies.KBHDP.utils import *
from watertap_contrib.reflo.analysis.case_studies.KBHDP.utils.ZLD_sweep_plot_functions.case_study_plotting import *
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import griddata

def create_contour_plot(df, x_var, y_var, z_var, title=None, save_path=None):
    """
    Create a contour plot from sweep results DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing sweep results
    x_var : str
        Column name for x-axis variable
    y_var : str
        Column name for y-axis variable  
    z_var : str
        Column name for z-axis (contour) variable
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the plot
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    # Extract data
    x = df[x_var].values
    y = df[y_var].values
    z = df[z_var].values

    # Calculate LCOE 
    z2 = (
        df["fs.energy.costing.total_capital_cost"] * df["fs.energy.costing.capital_recovery_factor"]
        + df["fs.energy.costing.total_operating_cost"]
    ) / df["fs.energy.cst.unit.heat_annual"]
    
    # Create grid for interpolation
    xi = np.linspace(x.min(), x.max(), 10)
    yi = np.linspace(y.min(), y.max(), 10)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate z values onto regular grid
    zi = griddata((x, y), z, (xi_grid, yi_grid), method='cubic')

    # Interpolate z values onto regular grid
    zi2 = griddata((x, y), z2, (xi_grid, yi_grid), method='cubic')

    # Create the plot
    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(16, 6))

    # Create contour plot
    contour = ax.contour(xi_grid, yi_grid, zi, levels=15, colors='black', alpha=0.6, linewidths=0.5)
    contourf = ax.contourf(xi_grid, yi_grid, zi, levels=15, cmap='viridis', alpha=0.8)


    # Create contour plot
    contour2 = ax1.contour(xi_grid, yi_grid, zi2, levels=15, colors='black', alpha=0.6, linewidths=0.5)
    contourf2 = ax1.contourf(xi_grid, yi_grid, zi2, levels=15, cmap='afmhot', alpha=0.8)

    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label("LCOT ($/m3)", rotation=270, labelpad=20)

    # Add colorbar
    cbar1 = plt.colorbar(contourf2, ax=ax1)
    cbar1.set_label("LCOE ($/kWh)", rotation=270, labelpad=20)
    
    # Add contour labels
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
    ax1.clabel(contour2, inline=True, fontsize=8, fmt='%.2f')
    # Set labels and title
    ax.set_xlabel("Cost per Total Aperture Area ($/m2)")
    ax.set_ylabel("Cost per Thermal Storage Capacity ($/kWh)")

    # Set labels and title
    ax1.set_xlabel("Cost per Total Aperture Area ($/m2)")
    ax1.set_ylabel("Cost per Thermal Storage Capacity ($/kWh)")
    
    # Tight layout
    plt.tight_layout()
    plt.show()

    return fig, ax
    

if __name__ == "__main__":

    plot_only = True

    sweep_dict = {
    'cst_cost_per_total_aperture_area':np.linspace(297*0.5, 297, 5),
    'cst_cost_per_storage_capital': np.linspace(62*0.5, 62, 5),
    } 
    
    input_dict = {
        'ro_water_recovery':0.8,
        'md_water_recovery':0.78,
        'nacl_recovery_price':0,
        'heat_price':0.00894,
        'electricity_price':0.04989,
        'grid_frac_heat': 0.5,
        'cst_cost_per_total_aperture_area': 297,
        'cst_cost_per_storage_capital': 62,
        'pv_cost_per_watt_installed': 1.6
    }


    #############################################################################################
    # Select sweep type
    #############################################################################################
    

    xcol_dict = {
        "ro_water_recovery":"fs.treatment.ro_water_recovery",
        "md_water_recovery":"water_recovery",
        "nacl_recovery_price":"fs.treatment.costing.nacl_recovered.cost",
        'cst_cost_per_total_aperture_area':'fs.energy.costing.trough_surrogate.cost_per_total_aperture_area',
        'cst_cost_per_storage_capital':'fs.energy.costing.trough_surrogate.cost_per_storage_capital',
        'pv_cost_per_watt_installed':'fs.energy.costing.pv_surrogate.cost_per_watt_installed',
        'grid_frac_heat':'fs.costing.frac_elec_from_grid'
    }

    ax_dict = {
        "ro_water_recovery": "RO Water Recovery (%)",
        "md_water_recovery": "MD Water Recovery (%)",
        "nacl_recovery_price": "NaCl Recovery Price ($/kg)",
        'cst_cost_per_total_aperture_area':"Cost per Total Aperture Area ($/m2)",
        'cst_cost_per_storage_capital':"Cost per Thermal Storage Capacity ($/kWh)",
        'pv_cost_per_watt_installed':"Cost per Watt Installed ($/W)",
        'grid_frac_heat':'Grid Fraction'
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
 
    if plot_only == False:
        m = zld_main(
                ro_recovery= input_dict['ro_water_recovery'],
                md_water_recovery= input_dict['md_water_recovery'],
                nacl_recovery_price = input_dict['nacl_recovery_price'],
                heat_price=input_dict['heat_price'],
                electricity_price=input_dict['electricity_price'],
                grid_frac_heat = input_dict["grid_frac_heat"],
                cost_per_total_aperture_area=input_dict['cst_cost_per_total_aperture_area'],
                cost_per_storage_capital=input_dict['cst_cost_per_storage_capital'],
                cost_per_watt_installed = input_dict['pv_cost_per_watt_installed'],
                )
        
        results_dict_test = build_results_dict(m, skips=skips)

        
        for i in sweep_dict["cst_cost_per_total_aperture_area"]:
            for j in sweep_dict["cst_cost_per_storage_capital"]:
                input_dict["cst_cost_per_total_aperture_area"] = i
                input_dict["cst_cost_per_storage_capital"] = j

                m = zld_main(
                    ro_recovery= input_dict['ro_water_recovery'],
                    md_water_recovery = input_dict['md_water_recovery'],
                    nacl_recovery_price = input_dict['nacl_recovery_price'],
                    heat_price=input_dict['heat_price'],
                    electricity_price=input_dict['electricity_price'],
                    grid_frac_heat = input_dict["grid_frac_heat"],
                    cost_per_total_aperture_area=input_dict['cst_cost_per_total_aperture_area'],
                    cost_per_storage_capital=input_dict['cst_cost_per_storage_capital'],
                    cost_per_watt_installed = input_dict['pv_cost_per_watt_installed'],
                    )
                
                results_dict_test = results_dict_append(m, results_dict_test)

        try:
            df = pd.DataFrame.from_dict(results_dict_test)
        except ValueError:
            df = pd.DataFrame.from_dict(results_dict_test, orient='index')


        filename = "/Users/mhardika/Documents/watertap-seto/Mukta-Work/kbhdp-case-study-md/ZLD_sweep_results/kbhdp_ZLD_cst_costs_sweep.csv"
        df.to_csv(filename)
    


    filename = "/Users/mhardika/Documents/watertap-seto/Mukta-Work/kbhdp-case-study-md/ZLD_sweep_results/kbhdp_ZLD_cst_costs_sweep.csv"
    df = pd.read_csv(filename, index_col=0)
    # Create contour plots
    x_var = 'fs.energy.costing.trough_surrogate.cost_per_total_aperture_area'
    y_var = 'fs.energy.costing.trough_surrogate.cost_per_storage_capital'
    z_var = 'fs.costing.LCOT'  # Adjust this to your actual LCOW column name
    
    # Plot 1: LCOW contour
    fig1, ax1 = create_contour_plot(
        df, 
        x_var=x_var,
        y_var=y_var, 
        z_var=z_var,
        title='LCOW Contour Plot: CST Costs Sensitivity',
    )