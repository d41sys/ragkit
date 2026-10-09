# Renewable Curtailment Mitigation via Locational Flexibility

Renewable curtailment occurs when generation from wind and solar farms is intentionally restricted by system operators due to transmission bottlenecks, localized overgeneration, or minimum operating constraints on thermal units required for system inertia. Curtailment represents both wasted clean energy and lost revenue for asset owners.

## Root Causes of Transmission Curtailment

Curtailment stems primarily from spatial mismatch between resource-rich generation corridors and distant demand load centers:
1. Thermal Transmission Bottlenecks: Overhead transmission lines have seasonal thermal MVA ratings. When generation exceeds transmission transfer capability, local assets are ordered to curtail.
2. Voltage and Angular Stability: Excessive power injections at weak peripheral points of the grid can cause localized overvoltage or exceed transient stability margins.
3. Minimum Inertia Constraints: Conventional synchronous plants must remain online at minimum technical stable load (P_min) to provide short-circuit capacity and inertia, forcing renewable curtailment even when renewables could technically satisfy 100% of demand.

## Locational Flexibility and Remedial Dispatch

Mitigating curtailment requires localized flexibility deployments:
- Co-Located Energy Storage: Placing BESS directly behind the substation constraint allows batteries to charge with energy that would otherwise be curtailed, discharging once line loading drops.
- Dynamic Line Rating (DLR): Installing environmental sensor packages on transmission lines to calculate real-time conductor cooling from wind and ambient temperature. DLR frequently reveals 10% to 30% additional line capacity compared to static seasonal ratings during windy periods.
- Flexible Industrial Demands: Siting electrolyzers, data centers, and industrial power-to-heat facilities adjacent to transmission-constrained renewable hubs absorbs localized excess power without stressing the bulk transmission corridor.
## Grid-Enhancing Technologies and Remedial Action Schemes

Transmission utilities are increasingly deploying Grid-Enhancing Technologies (GETs) alongside flexible energy storage to unlock trapped renewable generation. Modular power flow controllers (MPFCs) inject series voltage into overloaded transmission paths, actively redirecting power flows toward underutilized parallel lines within the meshed network. Furthermore, automated Remedial Action Schemes (RAS) monitor grid topology in real time. If a critical transmission circuit trips, the RAS triggers instantaneous software-directed battery charging or controlled generator tripping within milliseconds, preventing line thermal damage while allowing the transmission system to operate safely at higher continuous transfer limits.
