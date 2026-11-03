# Results recorded from the development notebook (AID, seed 0, val 600 / test 2,400)

## Baseline (all 3,000 images)
| score | accuracy |
| --- | --- |
| presence only | 37.10 |
| localization only | 30.27 |
| product | 39.07 |

## Trigger signals for "look again" (test, 1,516 errors)
| signal | AUROC | error recall @10/20/30% trigger rate |
| --- | --- | --- |
| low confidence | 0.550 | 8.8 / 19.8 / 31.5 |
| small top-2 margin | 0.669 | 12.1 / 23.8 / 35.9 |
| presence-top1 != loc-top1 | 0.672 | 13.4 / 27.0 / 39.6 |
| random | 0.449 | 8.6 / 17.8 / 26.8 |

Error types (test): not seen 653 (43%), ambiguous 490 (32%), other concept present 373 (25%).
Upper bounds: true class in presence top-3 for 31.9% of errors; top-5 oracle accuracy 65.5%.

## Area-based "main subject" verifier (negative result, test)
| rule (candidates M) | accuracy | fixed / broken |
| --- | --- | --- |
| presence (B0) | 36.8 | – |
| presence x sem_cov (3) | 34.5 | +176 / -232 |
| sem_cov only (3) | 27.1 | +236 / -470 |
| presence x inst_area (3) | 36.5 | +107 / -115 |
| best val-tuned mix (5, a=0.9) | 37.2 | – |
| margin-triggered, 30% rate | 37.4 | +59 / -46 (random 37.0) |

## Prompt test on 8 hard classes (val-selected, applied to test)
36.8 -> 42.5 (+201 / -66). Port 0 -> 77.5, Playground 1.2 -> 85.0, Desert 36.2 -> 70.0,
Commercial 0 -> 43.8, RailwayStation 2.5 -> 12.5, Church / School / Viaduct stay 0.

## Class skew (test, 80 true images per class)
Original names: Parking 408, BareLand 278, Farmland 234, Forest 154, Pond 151 predictions;
top-5 share 51% (uniform 17%). Never predicted: Church, Commercial, School, Viaduct.

## Full pipeline (test)
| method | accuracy | fixed / broken vs baseline |
| --- | --- | --- |
| original names | 36.8 | – |
| + PPR (T=2.0, chosen on val) | 44.5 | +303 / -119 |
| OCT-S2 (label-free selection) | 44.8 | +472 / -281 |
| OCT-S1 (val selection) | 51.4 | +435 / -85 |
| OCT-S1 + PPR | 56.3 | – |
| MPV (r=2, mean) | 44.1 | – |
| CLEAR = OCT + MPV + PPR | 58.7 | – |
| control: LR on 30 class-name presences | 77.1 (+PPR 76.7) | – |
| PSC: LR on 149 phrase presences | 85.4 | – |
| CLEAR-FS = PSC + PPR | 86.1 | – |
