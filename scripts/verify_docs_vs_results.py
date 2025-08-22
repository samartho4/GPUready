#!/usr/bin/env python3
import re, sys, json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
results_dir = root / 'results'
readme = root / 'README.md'
status = root / 'PROJECT_STATUS.md'
metrics_csv = results_dir / 'comprehensive_metrics.csv'
summary_md = results_dir / 'comprehensive_comparison_summary.md'
calib_md = results_dir / 'bnode_calibration_report.md'

issues = []

# Extract key numbers from summary
summary = summary_md.read_text() if summary_md.exists() else ''
rx_val = lambda k: re.search(k, summary)
# From summary, we record the top-level aggregates (already written in summary)

# Extract RMSE/R2 from README and STATUS
rd = readme.read_text() if readme.exists() else ''
st = status.read_text() if status.exists() else ''

# Assert no disallowed claims
bad_claims = [
    r"well-?calibrated",
    r"UDE\s*RMSE\s*:?\s*x1\s*:?\s*0\.0?23",
    r"UDE\s*RMSE\s*:?\s*x2\s*:?\s*0\.0?45",
]
for pat in bad_claims:
    if re.search(pat, rd, re.I) or re.search(pat, st, re.I):
        issues.append(f"Disallowed claim present matching /{pat}/")

# Check that timings appear consistently
ok_timing = re.search(r"UDE\s+.*?30\s*hours", rd, re.I) and re.search(r"30\s*hours", st, re.I)
if not ok_timing:
    issues.append("Timing mismatch: ensure UDE ~30 hours is documented in README and PROJECT_STATUS")

# Check that performance block exists
perf_ok = re.search(r"Physics.*RMSE x1.*0\.10", rd) and re.search(r"UDE.*RMSE x1.*0\.10", rd)
if not perf_ok:
    issues.append("Performance summary missing or malformed in README")

print(json.dumps({"ok": len(issues)==0, "issues": issues}, indent=2))
