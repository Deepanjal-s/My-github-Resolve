"""Public CPU notebook: Solar Cycle 25 from SILSO sunspot numbers, with a backtested end-of-cycle forecast.

    python build_notebook.py            # writes notebook/solar-cycle-25.ipynb + kernel-metadata.json
    python build_notebook.py --run CSV  # runs the code cells locally on a downloaded SN_m_tot_V2.0.csv
"""
import json
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# Solar Cycle 25 peaked higher than forecast. When will it end?

The Sun's activity rises and falls in a roughly 11-year cycle, counted in sunspots. Cycle 25 began in
December 2019. The 2019 NOAA/NASA prediction panel expected a weak cycle peaking around 115. This notebook
uses the official monthly sunspot numbers from SILSO (Royal Observatory of Belgium) to:

1. find every cycle since 1755 and where Cycle 25 actually peaked,
2. fit the standard Hathaway cycle shape to each cycle,
3. **backtest** three ways of forecasting when a cycle ends, on cycles the method never saw,
4. forecast the end of Cycle 25 (the start of Cycle 26) with an error range taken from the backtest.

Data: SILSO, "Monthly mean total sunspot number" (version 2.0), CC BY-NC 4.0, downloaded fresh each run.
Source: WDC-SILSO, Royal Observatory of Belgium, Brussels."""),
    code("""import io, os, urllib.request, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")

URL = "https://www.sidc.be/SILSO/INFO/snmtotcsv.php"
src = os.environ.get("SILSO_CSV")
raw = open(src, "rb").read() if src else urllib.request.urlopen(URL, timeout=60).read()
d = pd.read_csv(io.BytesIO(raw), sep=";", header=None, names=["year", "month", "t", "sn", "sd", "nobs", "provisional"])
d = d[d.sn >= 0]
t, sn = d.t.values, d.sn.values

# SILSO's 13-month smoothing: a centred window with half weight on the two end months.
w = np.r_[0.5, np.ones(11), 0.5] / 12
sm = np.full(len(sn), np.nan); sm[6:-6] = np.convolve(sn, w, mode="valid")
print(f"{len(d)} months, {d.year.iloc[0]}-{d.year.iloc[-1]}; last month {d.year.iloc[-1]}-{d.month.iloc[-1]:02d}: {sn[-1]}")"""),
    md("""## 1. Cycles since 1755

A cycle minimum is the lowest smoothed value within ±5 years, at least 8 years after the previous one."""),
    code("""mins = []
for i in range(6, len(sm) - 6):
    lo, hi = max(0, i - 60), min(len(sm), i + 61)
    if sm[i] == np.nanmin(sm[lo:hi]) and (not mins or t[i] - t[mins[-1]] > 8):
        mins.append(i)
peaks = []
for k, a in enumerate(mins):
    b = mins[k + 1] if k + 1 < len(mins) else len(sm) - 6
    peaks.append(a + int(np.nanargmax(sm[a:b])))
lengths = np.array([t[mins[k + 1]] - t[mins[k]] for k in range(len(mins) - 1)])
amp = np.array([sm[p] for p in peaks])
n = len(mins)
print(f"{n} cycles; cycle {n} started {t[mins[-1]]:.2f}")
print(f"cycle {n} smoothed peak: {amp[-1]:.1f} at {t[peaks[-1]]:.2f}  (rank {int((amp > amp[-1]).sum()) + 1} of {n} by height)")
print(f"completed cycles: length {lengths.mean():.2f} ± {lengths.std():.2f} years (range {lengths.min():.1f}-{lengths.max():.1f})")

fig, ax = plt.subplots(figsize=(11, 3.4))
ax.plot(t, sn, color="#c8ced6", lw=.6, label="monthly")
ax.plot(t, sm, color="#d9822b", lw=1.4, label="13-month smoothed")
ax.plot(t[mins], sm[mins], "v", color="#3b6fd8", ms=4, label="cycle minima")
for k, p in enumerate(peaks):
    ax.text(t[p], amp[k] + 12, str(k + 1), ha="center", fontsize=7)
ax.set_xlim(1750, t[-1] + 2); ax.set_ylabel("sunspot number"); ax.legend(frameon=False, ncol=3, loc="upper left")
ax.set_title("Sunspot cycles 1-25 (SILSO v2.0)"); plt.tight_layout(); plt.show()"""),
    md("""## 2. Where Cycle 25 stands

The panel forecast (about 115) is drawn for reference. Cycle 25 overshot it clearly, but it is a middling
cycle by historical standards."""),
    code("""a = mins[-1]
fig, ax = plt.subplots(figsize=(8, 3.4))
for k in range(n - 1):
    s, e = mins[k], mins[k + 1]
    ax.plot(t[s:e] - t[s], sm[s:e], color="#c8ced6", lw=.8)
ax.plot(t[a:] - t[a], sm[a:], color="#d9822b", lw=2.2, label="Cycle 25")
ax.axhline(115, ls="--", color="#555", lw=.9, label="2019 panel forecast peak (~115)")
ax.set_xlabel("years since cycle minimum"); ax.set_ylabel("smoothed sunspot number")
ax.set_title("Cycle 25 against cycles 1-24"); ax.legend(frameon=False); plt.tight_layout(); plt.show()"""),
    md("""## 3. Forecasting the end of a cycle, and checking the forecast honestly

The Hathaway et al. (1994) shape describes a cycle with three numbers:
`f(t) = a (t - t0)^3 / (exp(((t - t0)/b)^2) - 0.71)`.

To forecast a cycle's end from **the data available today** (Cycle 25 is 2 years past its peak), each past
cycle is fitted using only its first months up to 2 years after its peak. The forecast end is when the fitted
curve falls below a fraction of its maximum. That fraction, and the bias correction, are tuned on cycles 1-17
only. Cycles 18-24 are the held-out test.

Three forecasters, compared on the same held-out cycles:
- **calendar**: previous minimum + mean length of cycles 1-17,
- **amplitude regression**: length predicted from peak height (stronger cycles tend to be shorter),
- **curve fit (bias-corrected)**: the Hathaway fit above, with its average training error removed."""),
    code("""C = 0.71
def hathaway(x, a, b, t0):
    u = np.clip(x - t0, 1e-6, None)
    return a * u ** 3 / (np.exp((u / b) ** 2) - C)

HORIZON = 2.0
def fit_cycle(k):
    s, p = mins[k], peaks[k]
    cut = np.searchsorted(t, t[p] + HORIZON)
    x, y = t[s:cut], sn[s:cut]
    guess = [1.5 * amp[k] / max(1.0, t[p] - t[s]) ** 3, 4.0, t[s] - 0.5]
    return curve_fit(hathaway, x, y, p0=guess, maxfev=20000)[0]

def fit_end(params, frac, tpk):
    xs = np.arange(tpk, tpk + 15, 1 / 120); f = hathaway(xs, *params)
    return xs[np.argmax(f < frac * f.max())]

fits = [fit_cycle(k) for k in range(n)]
train, test = range(0, 17), range(17, n - 1)
frac = min(np.arange(.02, .4, .01), key=lambda f: np.mean([abs(fit_end(fits[k], f, t[peaks[k]]) - t[mins[k + 1]]) for k in train]))
bias = np.mean([fit_end(fits[k], frac, t[peaks[k]]) - t[mins[k + 1]] for k in train])
coef = np.polyfit(amp[:17], lengths[:17], 1)

def forecasts(k):
    return {"calendar": t[mins[k]] + lengths[:17].mean(),
            "amplitude regression": t[mins[k]] + np.polyval(coef, amp[k]),
            "curve fit (bias-corrected)": fit_end(fits[k], frac, t[peaks[k]]) - bias}

errors = {m: np.array([forecasts(k)[m] - t[mins[k + 1]] for k in test]) for m in forecasts(0)}
print(f"tuned on cycles 1-17: threshold {frac:.2f} of fitted max, bias {bias:+.2f} yr")
print(f"held-out cycles {test.start + 1}-{test.stop}: error = forecast - actual minimum (years)")
for m, e in errors.items():
    print(f"  {m:28s} mean abs error {np.abs(e).mean():.2f}   errors {np.round(e, 2)}")

fig, ax = plt.subplots(figsize=(8, 3.2))
xs = np.arange(len(test)); wdt = .27
for i, (m, e) in enumerate(errors.items()):
    ax.bar(xs + (i - 1) * wdt, e, wdt, label=f"{m} (MAE {np.abs(e).mean():.2f})")
ax.axhline(0, color="#333", lw=.8); ax.set_xticks(xs); ax.set_xticklabels([f"cycle {k + 1}" for k in test], fontsize=8)
ax.set_ylabel("forecast - actual (years)"); ax.legend(frameon=False, fontsize=8); ax.set_title("Backtest on held-out cycles")
plt.tight_layout(); plt.show()"""),
    md("""## 4. Forecast: the end of Cycle 25

The range uses the best method's held-out errors (forecast − actual), so it is only as wide as the
backtest says, not a formal confidence interval. Seven test cycles is a small sample."""),
    code("""k = n - 1
f = forecasts(k)
best = min(errors, key=lambda m: np.abs(errors[m]).mean())
lo, hi = f[best] - errors[best].max(), f[best] - errors[best].min()
for m, v in f.items():
    print(f"{m:28s} Cycle 25 minimum ~ {v:.2f}")
print(f"\\nbest in backtest: {best}: {f[best]:.2f}, backtest range {lo:.2f}-{hi:.2f}")

s = mins[k]; xs = np.arange(t[s], t[s] + 13, 1 / 12)
fig, ax = plt.subplots(figsize=(8.5, 3.4))
ax.plot(t[s:], sn[s:], color="#c8ced6", lw=.7, label="monthly")
ax.plot(t[s:], sm[s:], color="#d9822b", lw=2, label="smoothed")
ax.plot(xs, hathaway(xs, *fits[k]), color="#3b6fd8", lw=1.3, label="Hathaway fit (data to now)")
ax.axvspan(lo, hi, color="#3b6fd8", alpha=.12, label="forecast minimum (backtest range)")
ax.axvline(f[best], color="#3b6fd8", ls="--", lw=1)
ax.set_xlim(t[s], t[s] + 13); ax.set_ylabel("sunspot number"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Cycle 25 so far, and when Cycle 26 should begin"); plt.tight_layout(); plt.show()"""),
    md("""## 5. Bonus check: the Waldmeier effect

Stronger cycles rise to their peak faster. Does Cycle 25 fit the pattern?"""),
    code("""rise = np.array([t[peaks[k]] - t[mins[k]] for k in range(n)])
r = np.corrcoef(rise[:-1], amp[:-1])[0, 1]
print(f"correlation of rise time and peak height, cycles 1-24: {r:.2f}; Cycle 25 rose in {rise[-1]:.1f} yr to {amp[-1]:.0f}")
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.scatter(rise[:-1], amp[:-1], color="#9aa5b1", s=18, label="cycles 1-24")
ax.scatter(rise[-1:], amp[-1:], color="#d9822b", s=40, label="Cycle 25")
ax.set_xlabel("rise time (years)"); ax.set_ylabel("peak (smoothed)"); ax.legend(frameon=False)
ax.set_title(f"Waldmeier effect (r = {r:.2f})"); plt.tight_layout(); plt.show()"""),
    md("""## Takeaways

- Cycle 25 peaked well above the 2019 panel forecast, but by historical standards it is an ordinary cycle.
- A plain curve fit calls cycle ends too early. Once that bias is learned from old cycles, it beat the
  calendar and the amplitude regression on the seven held-out cycles.
- All three forecasters put the start of Cycle 26 around 2030-2031. The backtest range is the honest
  uncertainty; it will tighten as Cycle 25 declines, and this notebook updates itself with the data.

Notes: the threshold and bias are fitted on cycles 1-17 only; early cycles (before ~1850) are less reliable
data. Written with AI assistance; every number is computed in this notebook from the SILSO file."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = HERE / "notebook"
out.mkdir(exist_ok=True)
nbf.write(nb, out / "solar-cycle-25.ipynb")
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": "deepanjalshukla95/solar-cycle-25-when-will-it-end",
    "title": "Solar Cycle 25: when will it end?",
    "code_file": "solar-cycle-25.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": False, "enable_gpu": False, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
    "keywords": ["astronomy", "time series analysis"],
}, indent=1))
print("wrote", out / "solar-cycle-25.ipynb")

if "--run" in sys.argv:
    import os
    import matplotlib
    matplotlib.use("Agg")
    os.environ["SILSO_CSV"] = sys.argv[sys.argv.index("--run") + 1]
    ns = {}
    for c in cells:
        if c.cell_type == "code":
            exec(c.source, ns)
