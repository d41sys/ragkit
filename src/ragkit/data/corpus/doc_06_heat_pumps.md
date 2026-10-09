# Heat Pump Fleet Flexibility and Thermal Inertia Exploitation

Decarbonization of residential and commercial space heating is driving the electrification of heating loads through air-source and ground-source heat pumps. When aggregated, heat pump fleets represent gigawatts of shiftable electric capacity that can provide ancillary grid services without compromising occupant thermal comfort.

## Thermodynamic Principles and Coefficient of Performance (COP)

Heat pumps extract low-grade thermal energy from the ambient environment and transfer it to indoor spaces via a vapor-compression refrigeration cycle. The operational efficiency is quantified by the Coefficient of Performance (COP):
COP = Q_thermal / W_electrical.

COP varies continuously as a function of the temperature lift between the heat source (ambient air or ground loop) and the heat sink (radiators or underfloor hydronic piping). In sub-zero ambient conditions, air-source heat pump COP declines, necessitating higher electric power consumption for identical thermal output.

## Thermal Inertia and Virtual Storage Dispatch

The core flexibility resource of building heating systems is structural thermal inertia. Concrete floors, building envelopes, and hydronic buffer tanks act as virtual thermal energy storage. By pre-heating thermal masses by 1 to 2 degrees Celsius during periods of low wholesale electricity prices or renewable surplus, heat pumps can be turned down or shut off entirely for 2 to 4 hours during peak grid demand windows.

Aggregators deploy model predictive control (MPC) frameworks utilizing thermal resistor-capacitor (RC) network building models. MPC optimizes the compressor speed and thermal storage state across a 24-hour horizon while guaranteeing that room temperatures remain strictly bounded within user comfort bands (e.g., 20°C to 22°C). Communication protocols like SG Ready and EEBUS allow remote dispatch commands to modulate compressor cycles without voiding OEM manufacturer warranties.
## Refrigerant Selection and Domestic Hot Water Preheating

Thermodynamic optimization of heat pump fleets is further influenced by working fluids and secondary thermal circuits. Modern residential heat pumps are transitioning to low-GWP (Global Warming Potential) natural refrigerants such as R290 (propane) and CO2 (R744), which provide high supply temperatures (up to 75°C) suitable for legacy radiator retrofits without auxiliary electric resistance heating. Dedicated domestic hot water (DHW) storage tanks provide additional high-temperature flexibility: aggregators overheat DHW cylinders to 65°C during negative price periods, buffering both thermal energy and sanitizing tanks against Legionella bacteria while suppressing electricity consumption during evening grid peaks.
