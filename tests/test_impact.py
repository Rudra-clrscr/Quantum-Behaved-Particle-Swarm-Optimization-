"""
test_impact.py
--------------
Tests for the real-world impact conversions used by `scripts/impact_report.py`
and `data/impact_report.md`.

Runs offline. No network, no solver.

Note: this file originally also asserted that a frontend dashboard's copy of
these conversion factors matched `app/core/impact.py` (guarding against two
different CO2 factors being hardcoded in two places). That check was dropped
along with the frontend when this repo was trimmed down from MargdarshaQ to a
standalone algorithmic artifact — it no longer applies here.

Run with:  python test_impact.py
"""

import sys

from app.core.impact import (
    ASSUMPTIONS, ImpactAssumptions, compute_impact, project_to_city,
)

failures = []


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not condition:
        failures.append(label)


print("=" * 74)
print("1. Conversions are arithmetically right")
print("=" * 74)

# 80 km at 8 km/L = 10 L; 10 L x 2.68 = 26.8 kg CO2
r = compute_impact(km_saved=80.0, minutes_saved=120.0, deliveries=40)
check("litres = km / km_per_litre", abs(r.litres_saved - 10.0) < 1e-6, f"{r.litres_saved}")
check("CO2 = litres x factor", abs(r.kg_co2_saved - 26.8) < 1e-6, f"{r.kg_co2_saved}")
check("driver hours = minutes / 60", abs(r.driver_hours_saved - 2.0) < 1e-6, f"{r.driver_hours_saved}")
check("per-100 scales from the delivery count",
      abs(r.litres_per_100_deliveries - 25.0) < 1e-6, f"{r.litres_per_100_deliveries}")
check("assumptions travel with the result", "km per litre" in r.assumptions_note)

print()
print("=" * 74)
print("2. Nothing is clamped — a worse result stays negative")
print("=" * 74)

neg = compute_impact(km_saved=-108.6, minutes_saved=346.2, deliveries=40)
check("negative distance stays negative", neg.km_saved < 0, f"{neg.km_saved}")
check("negative distance means negative fuel", neg.litres_saved < 0, f"{neg.litres_saved}")
check("negative distance means negative CO2", neg.kg_co2_saved < 0, f"{neg.kg_co2_saved}")
check("a time saving alongside it stays positive", neg.driver_hours_saved > 0,
      f"{neg.driver_hours_saved}")

print()
print("=" * 74)
print("3. City projection refuses to guess the fleet size")
print("=" * 74)

p = project_to_city(r, vehicles_in_run=4, fleet_size=100)
# 100/4 = 25x, over 250 working days = 6250x
check("scales by fleet ratio x working days",
      abs(p.annual_litres - 10.0 * 6250) < 1.0, f"{p.annual_litres}")
check("tonnes derived from kg", abs(p.annual_tonnes_co2 - p.annual_kg_co2 / 1000) < 0.01)
check("labelled a projection", "not a measurement" in p.caveat)

for bad in (0, -1):
    try:
        project_to_city(r, vehicles_in_run=bad)
        check(f"vehicles_in_run={bad} raises", False, "no exception")
    except ValueError:
        check(f"vehicles_in_run={bad} raises rather than inflating", True)

print()
print("=" * 74)
print("4. Changing an assumption changes every derived figure")
print("=" * 74)

thirsty = ImpactAssumptions(km_per_litre=4.0)     # half the economy
half = compute_impact(80.0, 120.0, 40, assumptions=thirsty)
check("halving fuel economy doubles the litres",
      abs(half.litres_saved - 2 * r.litres_saved) < 1e-6,
      f"{r.litres_saved} -> {half.litres_saved}")
check("CO2 follows fuel", abs(half.kg_co2_saved - 2 * r.kg_co2_saved) < 1e-6)
check("the note reflects the changed assumption", "4 km per litre" in half.assumptions_note)

print()
print("=" * 74)
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1)
print("ALL CHECKS PASSED")
