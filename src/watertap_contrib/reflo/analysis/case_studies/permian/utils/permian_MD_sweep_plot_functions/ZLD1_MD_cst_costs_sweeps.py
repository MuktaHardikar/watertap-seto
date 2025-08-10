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

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import griddata

def create_heatmap(df, x_var, y_var, z_var, title=None, save_path=None, figsize=(10, 8)):
    """
    Create a heatmap from sweep results DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing sweep results
    x_var : str
        Column name for x-axis variable
    y_var : str
        Column name for y-axis variable  
    z_var : str
        Column name for z-axis (color) variable
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the plot
    figsize : tuple, optional
        Figure size (width, height)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    
    # Extract data
    x = df[x_var].values
    y = df[y_var].values
    z = df[z_var].values
    
    # Remove any NaN values
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x = x[mask]
    y = y[mask]
    z = z[mask]
    
    # Create pivot table for heatmap
    # First, let's see if we have a regular grid
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    
    print(f"Unique x values: {len(x_unique)}")
    print(f"Unique y values: {len(y_unique)}")
    print(f"Total points: {len(x)}")
    
    # Create a regular grid if we don't have one
    if len(x) == len(x_unique) * len(y_unique):
        # We have a complete grid
        z_matrix = z.reshape(len(y_unique), len(x_unique))
        x_grid = x_unique
        y_grid = y_unique
    else:
        # Interpolate to create a regular grid
        print("Creating interpolated grid...")
        xi = np.linspace(x.min(), x.max(), 10)
        yi = np.linspace(y.min(), y.max(), 10)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        z_interpolated = griddata((x, y), z, (xi_grid, yi_grid), method='linear')
        
        x_grid = xi
        y_grid = yi
        z_matrix = z_interpolated
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap using imshow for better control
    im = ax.imshow(z_matrix, cmap='viridis', aspect='auto', origin='lower',
                   extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(z_var, rotation=270, labelpad=20)
    
    # Overlay original data points
    scatter = ax.scatter(x, y, c='white', s=50, edgecolors='black', linewidth=1, alpha=0.8)
    
    # Add text annotations for values
    for i in range(len(x)):
        ax.annotate(f'{z[i]:.2f}', (x[i], y[i]), 
                   xytext=(0, 0), textcoords='offset points',
                   ha='center', va='center', fontsize=8, 
                   color='white', fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    
    if title is None:
        title = f'{z_var} Heatmap'
    ax.set_title(title)
    
    # Set ticks to show actual values
    ax.set_xticks(x_unique[::max(1, len(x_unique)//5)])  # Show max 5 ticks
    ax.set_yticks(y_unique[::max(1, len(y_unique)//5)])  # Show max 5 ticks
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

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
        'Qin': 5, 
        'tds': 130,
        'water_recovery':0.5,
        'grid_frac_heat':0.5,
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

    if plot_only == False:
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

            
    
        for i in sweep_dict["cst_cost_per_total_aperture_area"]:
            for j in sweep_dict["cst_cost_per_storage_capital"]:
                input_dict["cst_cost_per_total_aperture_area"] = i
                input_dict["cst_cost_per_storage_capital"] = j


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
                    
        try:
            df = pd.DataFrame.from_dict(results_dict_test)
        except ValueError:
            df = pd.DataFrame.from_dict(results_dict_test, orient='index')

        
        filename = "/Users/mhardika/Documents/watertap-seto/Mukta-Work/permian-case-study-md/ST2_MD_sweep_results//permian_ZLD1_MD_cst_costs_sweep.csv"
        df.to_csv(filename)
    
    filename = "/Users/mhardika/Documents/watertap-seto/Mukta-Work/permian-case-study-md/ST2_MD_sweep_results//permian_ZLD1_MD_cst_costs_sweep.csv"
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
    

    