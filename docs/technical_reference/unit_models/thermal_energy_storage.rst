Thermal Energy Storage (TES)
====================================================

This is a thermal energy storage (TES) model assumes the tank is at a uniform temperature (similar to a continuous stirred tank).
This TES model supports supports steady-state only.

Model Structure
---------------

This TES model consists of 4 StateBlocks (as 4 Ports in parenthesis below). Two state blocks connect
to the the external heat exchanger which adds heat to the TES and two ports connect to the process side
and provide heat to the treatment process.

* Heat exchanger inlet (tes_hx_inlet)
* Heat exchanger outlet (tes_hx_outlet)
* Process inlet (tes_process_inlet)
* Process outlet (tes_process_outlet)

Sets
----

.. csv-table::
   :header: "Description", "Symbol", "Indices"

   "Time", ":math:`t`", "[0]"
   "Phases", ":math:`p`", "['Liq', 'Vap']"
   "Components", ":math:`j`", "['H2O']"

Degrees of Freedom/ Variables
------------------

The TES model has 4 degrees of freedom that should be fixed for the unit to be fully specified
in addition to the state variables at the inlet and outlet.
Typically the following variables define the input and output heat exchanger and process ports listed above. 

.. csv-table::
   :header: "Variables", "Variable name", "Symbol", "Valid range", "Unit"

   "Temperature", "temperature", ":math:`T_{f}`", "298 - 398", ":math:`\text{K}`"
   "Pressure", "pressure", ":math:`P`", "", ":math:`Pa`"
   "Mass flow rate of vapor phase", "flow_mass_phase_comp[0,'Vap','H2O']", "", ":math:`kg/s`"
   "Mass flow rate of liquid phase", "flow_mass_phase_comp[0,'Liq','H2O']", "", "", ":math:`kg/s`"
   
The following variables should also be fixed. An initial temperature is assigned to the outlet stream at the heat exhanger and process loop.

.. csv-table::
   :header: "Variables", "Variable name", "Symbol", "Valid range", "Unit"

   "Initial temperature", "tes_initial_temp", ":math:`T_{0}`", "298 - 398", ":math:`\text{K}`"
   "Time step", "dt", ":math:`d_{t}`", "", ":math:`h`"
   "Hours of storage", "hours_storage", "0-24", ":math:`h`"
   "Heat load", "heat_load", ":math:`\text{heat_load}`", "", ":math:`MWh`"
   

Variables
---------
The system configuration variables should be fixed at the default values, 
with which the surrogate model was developed:

.. csv-table::
   :header: "Description", "Symbol", "Variable Name", "Value", "Units"

   "Temperature difference between the last and first effect", ":math:`\Delta T_{last}`", "delta_T_last_effect", "10", ":math:`\text{K}`"
   "Temperature decrease in cooling reject water", ":math:`\Delta T_{cooling}`", "delta_T_cooling_reject", "-3", ":math:`\text{K}`"
   "System thermal loss faction", ":math:`f_{Q_{loss}}`", "thermal_loss", "0.054", ":math:`\text{dimensionless}`"

The following performance variables are derived from the surrogate equations:

.. csv-table::
   :header: "Description", "Symbol", "Variable Name", "Index", "Units"

   "Gain output ratio", ":math:`GOR`", "gain_output_ratio", "None", ":math:`\text{dimensionless}`"
   "Specific total area", ":math:`sA`", "specific_area_per_m3_day", "None", ":math:`\text{m}^2\text{ per m}^3\text{/day}`"

The following variables are calculated by fixing the default degree of freedoms above.

.. csv-table::
   :header: "Description", "Symbol", "Variable Name", "Units"

   "Thermal power requirement", ":math:`P_{req}`", "thermal_power_requirement",  ":math:`\text{kW}`"
   "Specific thermal energy consumption", ":math:`STEC`", "specific_energy_consumption_thermal",  ":math:`\text{kWh} / \text{m}^3`"
   "Total seawater mass flow rate (feed + cooling)", ":math:`m_{seawater,total}`", "feed_cool_mass_flow",  ":math:`\text{kg} / \text{s}`"
   "Total seawater volumetric flow rate (feed + cooling)", ":math:`v_{seawater,total}`", "feed_cool_vol_flow",  ":math:`\text{m}^3 / \text{h}`"


Equations
---------
.. csv-table::
   :header: "Description", "Equation"

   "Temperature in the last effect", ":math:`T_{last} = \Delta T_{last} + T_{feed}`"
   "Temperature of outlet cooling water", ":math:`T_{cooling,out} = \Delta T_{cooling,in} + T_{feed}`"
   "Distillate volumetric flow rate (production rate)", ":math:`v_{distillate} = v_{feed} T_{feed}`"
   "Steam mass flow rate", ":math:`m_{steam} = m_{distillate} / GOR`"
   "Specific thermal energy consumption", ":math:`STEC = \frac{\Delta H_{vap} \times \rho_{distillate}}{GOR}`"
   "Thermal power requirement", ":math:`P_{req} = STEC \times v_{distillate}`"
   "Energy balance", ":math:`v_{seawater,total} \times (H_{cooling} - H_{feed}) = (1 - f_{Q_{loss}})\times P_{req} - m_{brine} H_{brine} - m_{distillate} H_{distillate} + m_{feed} H_{cooling}`"

Surrogate equations and the corresponding coefficients for different number of effects can be found in the unit model class.

.. TODO: add link to the code of LT-MED unit model class

References
----------

[1] Palenzuela, P., Hassan, A. S., Zaragoza, G., & Alarcón-Padilla, D. C. (2014). Steady state model for
multi-effect distillation case study: Plataforma Solar de Almería MED pilot plant. Desalination, 337,
31-42.

[2] Ortega-Delgado, B., Garcia-Rodriguez, L., & Alarcón-Padilla, D. C. (2017). Opportunities of
improvement of the MED seawater desalination process by pretreatments allowing high-temperature
operation. Desalin Water Treat, 97, 94-108.