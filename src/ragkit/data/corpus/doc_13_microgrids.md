# Islanded and Grid-Connected Microgrid Operation

A microgrid is a localized group of electricity sources and loads that normally operates connected to the centralized electrical grid, but can disconnect and function autonomously in islanded mode to maintain critical service during utility outages.

## Interconnection Modes and Seamless Islanding

Microgrids operate across two distinct operational modes:
1. Grid-Connected Mode: The microgrid imports or exports power across the Point of Common Coupling (PCC). Local distributed energy resources optimize economic dispatch, peak shaving, and energy arbitrage while taking voltage and frequency reference from the stiff utility grid.
2. Islanded (Autonomous) Mode: When a utility outage occurs, static disconnect switches isolate the microgrid within tens of milliseconds. In islanded mode, the microgrid must maintain its own independent voltage and frequency stability without the inertial backing of utility generators.

## Control Hierarchy: Primary, Secondary, and Tertiary

Microgrid energy management systems (EMS) are structured into a three-tier control hierarchy:
- Primary Control: Millisecond-level inverter droop control and grid-forming voltage control, establishing power sharing among distributed inverters without inter-unit communications.
- Secondary Control: Second-to-minute level control that eliminates steady-state frequency and voltage deviations introduced by primary droop response, restoring grid variables to nominal 50/60 Hz and 400/480 V.
- Tertiary Control: Minute-to-hour level economic optimization coordinating power exchange at the PCC, managing fuel supplies for diesel/gas backup generators, and optimizing renewable utilization.
## Resynchronization and Black-Start Sequencing

Transitioning an islanded microgrid back to grid-connected operation requires precise electrical synchronization across the Point of Common Coupling (PCC). Microgrid synchronizers monitor voltage magnitude, frequency, and phase angle differences across the breaker. Grid-forming inverters gently adjust their internal frequency reference until phase angle error across the tie breaker falls below 5 degrees and voltage difference is within 2%, at which point the static transfer switch closes without causing damaging transient inrush currents. In the event of a total microgrid blackout, black-start logic sequentially re-energizes local generation, closes feeder circuits, and restores critical community loads in prioritized tranches.
