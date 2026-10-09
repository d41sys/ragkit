# Electric Vehicle Smart Charging and V1G/V2G Grid Integration

The global transition to electric mobility introduces substantial coincident load to low-voltage distribution grids. However, parked electric vehicles (EVs) with onboard battery capacities ranging from 40 kWh to 100 kWh represent flexible distributed energy resources through unidirectional smart charging (V1G) and bidirectional vehicle-to-grid (V2G) integration.

## V1G Unidirectional Controlled Charging

V1G refers to modulating the charging rate (amperage) and timing of an EV in response to external signals without back-feeding power into the grid. Controlled charging strategies shift charging cycles to overnight hours with low locational marginal prices or midday windows with abundant solar generation. Protocols such as ISO 15118 and OCPP 2.0.1 enable automated negotiation of charging profiles between Electric Vehicle Supply Equipment (EVSE), the vehicle's onboard battery management system (BMS), and charging station aggregators. Dynamic phase switching between single-phase and three-phase charging allows low-power trickle charging when solar production drops.

## Vehicle-to-Grid (V2G) and Vehicle-to-Building (V2B)

V2G transforms the EV into an active distributed storage asset capable of bidirectional power injection. Utilizing grid-forming or grid-following bidirectional inverters, V2G systems inject power during peak demand periods, provide synthetic inertia, and supply local emergency backup during grid outages (Vehicle-to-Home / V2H).

Key technical barriers to widespread V2G deployment include battery cycle degradation and regulatory grid interconnection codes (IEEE 1547 and UL 1741). High-frequency shallow micro-cycling during frequency regulation requires careful battery health monitoring to avoid cell degradation. OEM warranties increasingly include throughput-based or cumulative energy discharge clauses to govern commercial V2G participation.
## Fleet Telematics and High-Power DC Depots

Commercial fleet electrification (delivery vans, municipal buses, and heavy-duty logistics trucks) demands high-power Megawatt Charging Systems (MCS) exceeding 1 MW per vehicle. At logistics depots, uncoordinated charging of several dozen electric trucks simultaneously would severely breach local substation transformer ratings. Depots deploy centralized charging controllers running quadratic programming algorithms that ingest fleet telematics, route schedules, and battery state of charge. The system dynamically modulates DC charging dispensers to guarantee vehicles achieve target state of charge before their scheduled route departure times while keeping facility peak demand within contracted utility connection limits.
