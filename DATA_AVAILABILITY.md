# Data availability and input inventory

No observation table, derived data table, forecast snapshot, checkpoint, or historical result is distributed in this source-only release.

The manuscript acknowledges NOAA CO-OPS and ECMWF/ERA5 as data providers. That acknowledgement alone does not establish the complete provenance or redistribution rights of the locally assembled tables. The coastal-feature source and applicable terms also need to be documented. Data will not be relabeled, fabricated, or silently substituted.

## Required inputs retained in a separate local review snapshot

| Expected path | Local file count | Local bytes | Public release status |
| --- | ---: | ---: | --- |
| `data/processed/preexperiment/event_dataset.csv` | 1 | 9,993,701 | Not included; provenance and redistribution terms unverified |
| `data/processed/coastal_context/coastal_static_features.csv` | 1 | 2,759 | Not included; provenance and redistribution terms unverified |
| `results/handcrafted_seed_averaged/predictions/methods/` | 168 | 4,560,510 | Not included; historical inputs, protocol lineage unverified |
| `results/handcrafted_seed_averaged_ntr/predictions/methods/` | 168 | 4,561,794 | Not included; historical inputs, protocol lineage unverified |
| `results/backbone_followups/stgcn_lite_017i_phase_direction_bonus_b0p2_c0p5_r0p5_f0p05/predictions/` | 56 | 3,366,033 | Not included; exact generating source revision unestablished |

Counts and sizes are a read-only inventory, not a validation of the data or the historical calculations.

The event table schema includes station identifiers, public station names and coordinates, event identifiers, UTC timestamps, water levels, predicted tide, meteorological and wave variables, derived lag/rolling features, and feature-availability flags. The coastal table includes station coordinates and derived coastal context. No new data were downloaded or generated for this release.

## What is still needed

- Exact upstream product identifiers, retrieval instructions, time ranges, transformations, and applicable license/attribution terms for each table.
- Evidence identifying the code, training/selection protocol, and inputs that produced the frozen Stage-1 and graph forecasts.
- Required model checkpoints and a verified environment, if they can be lawfully distributed.
- A versioned mapping from the verified code and data to the manuscript figures and tables.

Until these items are resolved, this repository must not be cited as an archive of all data supporting a successful independent reproduction.
