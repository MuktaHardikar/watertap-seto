Chemical Softening
====================================================

This chemical softening unit model calculates the chemical dose required for target removal of hardness causing components. 
The model also calculates the size the mixer, flocculator, sedimentation basin and the recarbonation basin. This chemical softening model
   * supports steady-state only
   * predicts the outlet concentration of Ca2+ and Mg2+
   * is verified against literature data

Configuration Inputs
--------------------

The model requires 2 configuration inputs
   * Softening procedure: ``single_stage_lime``, ``excess_lime``, ``single_stage_lime_soda``, ``excess_lime_soda``
   * Silica removal: ``True``, ``False``

Degrees of Freedom/Variables
------------------

The chemical softening model has 18 degrees of freedom that should be fixed for the unit to be fully specified. 
Additionally, depending on the chemical softening process selected, chemical dosing may or may not be required to be fixed.

Typically, the following 7 variables define the input feed.

.. csv-table::
   :header: "Variables", "Variable Name",  "Unit"

   "Feed volume flow rate", "properties_in[0].flow_mass_phase_comp['Liq','H2O']",  ":math:`\text{m}^3 / \text{s}`"
   "Feed composition", "properties_in[0].flow_mass_phase_comp['Liq','Ca_2+']",  ":math:`\text{g/}\text{L}`"
   "Feed composition", "properties_in[0].flow_mass_phase_comp['Liq','Mg_2+']",  ":math:`\text{g/}\text{L}`"
   "Feed composition", "properties_in[0].flow_mass_phase_comp['Liq','Alkalinity_2-']",  ":math:`\text{g/}\text{L}`"
   "Feed temperature", "feed_props.temperature",  ":math:`^o\text{C}`"
   "Ca2+ effluent target in CaCO3 equivalents", "ca_eff_target", ":math:`\text{g/}\text{L}`"
   "Mg2+ effluent target in CaCO3 equivalents", "mg_eff_target", ":math:`\text{g/}\text{L}`"
   
The following 11 variables define the system design.

.. csv-table::
   :header: "Variables", "Variable Name", "Valid Range", "Unit"

   "Number of mixers", "no_of_mixer", "", ":math:`\text{dimensionless}`"
   "Number of flocculators", "no_of_floc","",  ":math:`\text{dimensionless}`"
   "Retention time of mixer", "retention_time_mixer", "0.1-5", ":math:`\text{min}`"
   "Retention time of flocculator", "retention_time_floc", "10-45", ":math:`\text{min}`"
   "Retention time of sedimentation basin", "retention_time_sed","120-240",  ":math:`\text{min}`"
   "Retention time of recarbonation basin", "retention_time_recarb", "15-30", ":math:`\text{min}`"
   "Fractional volume recovery", "frac_vol_recovery","", ":math:`\text{dimensionless}`"
   "Removal efficiency of components (except Ca2+ and Mg2+)", "removal_efficiency", "",":math:`\text{dimensionless}`"
   "CO2 dose in CaCO3 equivalents", "CO2_CaCO3","", ":math:`\text{g/}\text{L}`"
   "Velocity gradient in mixer", "vel_gradient_mix", "300-1000",":math:`\text{/}\text{s}`"
   "Velocity gradient in flocculator", "vel_gradient_floc","20-80", ":math:`\text{/}\text{s}`"

The following variables should be fixed to 0 if their dose is not calculated in the softening procedure for the model to be fully specified. 
The softening procedure where the doses are calculated in are listed in the table.

.. csv-table::
   :header: "Variables", "Softening procedure", "Variable Name",  "Unit"

   "Excess lime", "excess_lime, excess_lime_soda", "excess_CaO", ":math:`\text{g/}\text{L}`"
   "Soda ash","single_stage_lime_soda, excess_lime_soda ", "Na2CO3_dosing", ":math:`\text{g/}\text{L}`" 
   "CO2 dose in second basin","excess_lime_soda", "CO2_second_basin", ":math:`\text{g/}\text{L}`" 
   "MgCl2","Silica removal", "MgCl2_dosing", ":math:`\text{g/}\text{L}`" 


Solution Composition
---------------


Model Structure
---------------

This chemical softening model consists of 3 StateBlocks (as 3 Ports in parenthesis below).

* Inlet (inlet)
* Outlet (outlet)
* Waste (waste)

The softening procedure type and whether or not silica removal is desired is set up in the configuration of the unit block.

Sets
----

The components Ca2+, Mg2+ and Alkalinity_2- must be included in the components.

.. csv-table::
   :header: "Description", "Symbol", "Indices"

   "Time", ":math:`t`", "[0]"
   "Phases", ":math:`p`", "['Liq', 'Vap']"
   "Components", ":math:`j`", "['H2O', 'Ca_2+', ' Mg_2+', 'Alkalinity_2-']"

Parameters
---------

The following parameters are used as default values and are not mutable. 

.. csv-table::
   :header: "Description", "Parameter Name"

   "Ratio of MgCl2 to SiO2", "``MgCl2_SiO2_ratio``"
   "Sludge produced per kg Ca in CaCO3 hardness", "``Ca_hardness_CaCO3_sludge_factor``"
   "Sludge produced per kg Mg in CaCO3 hardness", "``Mg_hardness_CaCO3_sludge_factor``"
   "Sludge produced per kg Mg in non-CaCO3 hardness", "``Mg_hardness_nonCaCO3_sludge_prod_factor``"
   "Multiplication factor to calculate excess CaO", "``excess_CaO_coeff``"


Equations
---------

The chemical dose is calculated based on the type of softening procedure selected in the configuration of the flowsheet.

.. csv-table:: Single Stage Lime
   :header: "Description", "Equation"

   "Lime dose", "Carbonic acid concentration + Calcium carbonate hardness"
   "Soda ash dose", "None"
   "Carbon dioxide first stage", "Alkalinity - Calcium hardness + Residual calcium hardness"
 
.. csv-table:: Excess Lime
   :header: "Description", "Equation"

   "Lime dose", "Carbonic acid concentration + Total alkalinity + Magnesium hardness + Excess lime dose"
   "Soda ash dose", "None"
   "Carbon dioxide first stage", "Alkalinity - Total hardness + Residual calcium hardness + Residual magnesium hardness"

.. csv-table:: Single Stage Lime-Soda Ash
   :header: "Description", "Equation"

   "Lime dose", "Carbonic acid concentration + Calcium carbonate hardness"
   "Soda ash dose", "Calcium non-carbonate hardness and/or Magnesium non-carbonate hardness"
   "Carbon dioxide first stage", "Alkalinity + Soda ash dose - Calcium hardness + Residual calcium hardness"

.. csv-table:: Excess Lime-Soda Ash
   :header: "Description", "Equation"

   "Lime dose", "Carbonic acid concentration + Calcium carbonate hardness + 2*Magnesium hardness + Magnesium non-carbonate hardness + Excess lime"
   "Soda ash dose", "Calcium non-carbonate hardness + Magnesium non-carbonate hardness"
   "Carbon dioxide first stage", "Lime dose + Residual magnesium hardness"
   "Carbon dioxide second stage", "Alkalinity + Soda ash dose - Source total hardness + Residual hardness"

The following equations are independent of the softening procedure selected but depend on the feed composition.

.. csv-table::
   :header: "Description", "Equation"

   "MgCl2 dose (if silica removal is selected)", "MgCl2_SiO2_ratio * properties_in[0].conc_mass_phase_comp['Liq', 'SiO2'] "
   "Sludge produced", "properties_in[0].flow_vol_phase['Liq'] * 
   
   (Ca_hardness_CaCO3_sludge_factor * Ca_hardness_CaCO3 + 
   
   Mg_hardness_CaCO3_sludge_factor * Mg_hardness_CaCO3 + 
   
   Ca_hardness_nonCaCO3 +  
   
   Mg_hardness_nonCaCO3_sludge_prod_factor * Mg_hardness_nonCaCO3 + 
   
   excess_CaO + prop_in.conc_mass_phase_comp['Liq', 'TSS'] + MgCl2_dosing)"
   "Volume of mixer", "properties_in[0].flow_vol_phase['Liq'] * retention_time_mixer * no_of_mixer"
   "Volume of flocculator", "properties_in[0].flow_vol_phase['Liq'] * retention_time_floc * no_of_floc"
   "Volume of sedimentation basin", "properties_in[0].flow_vol_phase['Liq'] * retention_time_sed"
   "Volume of recarbonation basin", "properties_in[0].flow_vol_phase['Liq'] * retention_time_recarb"

Costing
---------

The following table lists out the coefficients used in the equations to calculate the capital and operating costs
for the mixer, flocculator, sedimentation basin and recarbonation basin. The coefficients are assigned as mutable Parameters.

.. csv-table::
   :header: "Unit", "Symbol", "_constant", "_coeff/_coeff_1", "_coeff_2","_coeff_3","_exp/_exp_1","_exp_2"

   "Mixer", "mix_tank_capital", "28584", "0.0002","22.776","", "2", "" 
   "Flocculator", "floc_tank_capital", "217222", "673894", "", "", "", ""
   "Sedimentation basin", "sed_basin_capital", "182801", "-0.0005", "86.89", "", "2", ""
   "Recarbonation basin", "recarb_basin_capital", "19287", "4e-9", "-0.0002", "10.027", "3", "2"
   "Recarbonation basin source", "recarb_basin_source_capital", "130812", "9e-8", "-0.001", "42.578", "", "2"
   "Lime feed system", "lime_feed_system_capital", "193268", "20.065", "", "", "", ""
   "Administrative capital", "floc_tank_capital", "", "", "", "", "", ""
   "Mixer", "mix_tank_capital", "28584", "0.0002","22.776","", "2", "" 
   "Flocculator", "floc_tank_capital", "", "", "", "", "", ""
   "Sedimentation basin", "floc_tank_capital", "", "", "", "", "", ""
   "Recarbonation basin", "floc_tank_capital", "", "", "", "", "", ""

The following equations are used to calculate the capital and operating costs for the mixer, flocculator, sedimentation basin and recarbonation basin units
and other costs.

.. csv-table::
   :header: "Unit", "Symbol", "Equation"


The following equations are used to calculate the power consumption by the mixer and the flocculator used to calculate total electricity consumption

.. csv-table::
   :header: "Unit", "Symbol", "Equation"

References
----------

[1]  Crittenden, J. C., & Montgomery Watson Harza (Firm). (2012). Water treatment principles and design. Hoboken, N.J: J.Wiley.

[2]  Davis, M. L. (2010). Water and wastewater engineering: Design principles and practice.

[3]  Baruth. (2005). Water treatment plant design / American Water Works Association, American Society of Civil Engineers; Edward E. Baruth, technical editor. (Fourth edition.). McGraw-Hill.

[4]  Edzwald, J. K., & American Water Works Association. (2011). Water quality & treatment: A handbook on drinking water. New York: McGraw-Hill.

[5]  R.O. Mines Environmental Engineering: Principles and Practice, 1st Ed, John Wiley & Sons

[6]  Lee, C. C., & Lin, S. D. (2007). Handbook of environmental engineering calculations. New York: McGraw Hill.

