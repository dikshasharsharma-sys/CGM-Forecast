# Feature Harmonization Report

## AZT1D Raw Columns
- Basal
- BolusType
- CGM
- CarbSize
- CorrectionDelivered
- DeviceMode
- EventDateTime
- FoodDelivered
- Readings (CGM / BGM)
- TotalBolusInsulinDelivered

## HUPA Raw Columns
- basal_rate
- bolus_volume_delivered
- calories
- carb_input
- glucose
- heart_rate
- steps
- time

## Common Columns (After Standardization)
- basal_rate
- bolus_volume_delivered
- carb_input
- glucose
- subject_id
- time

## Dropped / Non-Common Columns
### AZT1D
- BolusType
- CorrectionDelivered
- DeviceMode
- FoodDelivered

### HUPA
- calories
- heart_rate
- steps

## Notes
- AZT1D glucose target is derived from the `CGM` column (fallback to `Readings (CGM / BGM)` when present).
- HUPA glucose target is `glucose` from the preprocessed files (semicolon-separated).

## Estimated CGM Sampling Interval (minutes)
- AZT1D: 5.00
- HUPA: 5.00

## Forecasting Horizons (steps ahead)
- AZT1D:
  - 30m: 6 steps
  - 60m: 12 steps
- HUPA:
  - 30m: 6 steps
  - 60m: 12 steps

## Lag Feature Configuration
- Common lag count used across datasets: 18