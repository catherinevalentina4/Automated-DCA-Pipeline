"""
Decline Curve Analysis - SINGLE WELL (generik, tinggal ganti FILE_PATH)

@author: Catherine Valentina 


CARA PAKAI (how to use)
----------
Cukup ubah FILE_PATH ke lokasi file Excel sumur yang ingin dianalisis
(kolom wajib: "Date" dan "Oil bbl/d"), lalu OUTPUT_PATH untuk hasilnya.
Tidak ada parameter lain yang perlu di-tuning per sumur.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit, brentq
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================================
# KONFIGURASI
# =====================================================================
# Bisa diisi lewat argumen command line:
#   python dca_single_well.py --file data/sample/well_sample.xlsx --output result.xlsx
# Kalau dijalankan langsung di Spyder/IDE tanpa argumen, dipakai nilai
# DEFAULT_FILE_PATH / DEFAULT_OUTPUT_PATH di bawah -- ganti sesuai lokasi
# file Anda sendiri (folder ini TIDAK ikut diupload ke GitHub, lihat .gitignore).

import argparse

DEFAULT_FILE_PATH = r"data/sample/well_sample.xlsx"
DEFAULT_OUTPUT_PATH = r"DCA_Result_sample.xlsx"

parser = argparse.ArgumentParser(description="Automated Decline Curve Analysis (single well)")
parser.add_argument("--file", dest="file_path", default=DEFAULT_FILE_PATH,
                     help="Path ke file Excel data produksi (kolom: Date, Oil bbl/d)")
parser.add_argument("--output", dest="output_path", default=DEFAULT_OUTPUT_PATH,
                     help="Path file Excel hasil analisis")
parser.add_argument("--economic-limit", type=float, default=5,
                     help="Economic limit rate (bbl/d), default 5")
# parse_known_args supaya tetap aman dijalankan di Spyder/Jupyter (yang suka
# menyisipkan argumen tambahan sendiri)
args, _ = parser.parse_known_args()

FILE_PATH = args.file_path
OUTPUT_PATH = args.output_path
ECONOMIC_LIMIT = args.economic_limit   # bbl/d - dipakai untuk shut-in detection & forecast

MIN_SEGMENT_LEN = 90        # panjang minimum segmen (hari)
MIN_SEGMENT_MERGE = 60      # segmen lebih pendek dari ini digabung ke sebelumnya
WORKOVER_THRESHOLD_PCT = 30 # persen kenaikan rate dianggap kemungkinan workover
MAX_FORECAST_DAYS = 365 * 30

pd.set_option("display.width", 150)


# =====================================================================
# 1. LOAD & CLEANING DATA
# =====================================================================

import os

if not os.path.exists(FILE_PATH):
    folder = os.path.dirname(FILE_PATH)
    msg = f"File tidak ditemukan:\n  {FILE_PATH}\n"
    if os.path.exists(folder):
        available = os.listdir(folder)
        msg += "\nFolder-nya ADA, tapi nama file di FILE_PATH tidak cocok.\n"
        msg += f"File yang tersedia di '{folder}':\n"
        for f in available:
            msg += f"  - {f}\n"
        msg += "\nCek lagi ejaan nama file (spasi, huruf besar/kecil, ekstensi) di FILE_PATH."
    else:
        msg += f"\nFolder-nya JUGA tidak ditemukan: {folder}\n"
        msg += "Cek lagi apakah folder sudah dibuat / sudah sinkron dari OneDrive."
    raise FileNotFoundError(msg)

df = pd.read_excel(FILE_PATH)
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
df = df.drop_duplicates()
df = df.dropna()
df = df[df["Oil bbl/d"] >= 0].reset_index(drop=True)
df["Time"] = (df["Date"] - df["Date"].min()).dt.days

last_date = df["Date"].max()
last_actual_rate = df["Oil bbl/d"].iloc[-1]

print("=" * 40)
print("Production Summary")
print("=" * 40)
print("Jumlah Data       :", len(df))
print("Tanggal Awal      :", df["Date"].min())
print("Tanggal Akhir     :", last_date)
print("Rate Terakhir     :", last_actual_rate, "bbl/d")
print("Produksi Maksimum :", df["Oil bbl/d"].max())
print("Produksi Rata-rata:", round(df["Oil bbl/d"].mean(), 2))

plt.figure(figsize=(14, 5))
plt.plot(df["Date"], df["Oil bbl/d"], color="blue")
plt.title("Oil Production History")
plt.xlabel("Date")
plt.ylabel("Oil Rate (bbl/d)")
plt.grid(True)
plt.show()


# =====================================================================
# 2-5. DATA QUALITY, OUTLIER, SHUT-IN, WORKOVER (informasional)
# =====================================================================

Q1 = df["Oil bbl/d"].quantile(0.25)
Q3 = df["Oil bbl/d"].quantile(0.75)
IQR = Q3 - Q1
df["Outlier"] = (df["Oil bbl/d"] < Q1 - 1.5 * IQR) | (df["Oil bbl/d"] > Q3 + 1.5 * IQR)
print("\nJumlah Outlier :", df["Outlier"].sum())

df["ShutIn"] = df["Oil bbl/d"] <= ECONOMIC_LIMIT

df["Rate Change"] = df["Oil bbl/d"].diff()
df["Rate Change %"] = df["Oil bbl/d"].pct_change() * 100
df["Rate Change %"] = df["Rate Change %"].replace([np.inf, -np.inf], np.nan)
df["Workover"] = df["Rate Change %"] > WORKOVER_THRESHOLD_PCT

plt.figure(figsize=(14, 5))
plt.plot(df["Date"], df["Oil bbl/d"], color="lightgray", label="Production")
plt.scatter(df.loc[df["ShutIn"], "Date"], df.loc[df["ShutIn"], "Oil bbl/d"],
            color="red", s=15, label="Shut-In")
plt.scatter(df.loc[df["Workover"], "Date"], df.loc[df["Workover"], "Oil bbl/d"],
            color="green", s=15, label="Possible Workover")
plt.title("Shut-In & Workover Detection")
plt.legend()
plt.grid()
plt.show()


# =====================================================================
# 6. AUTOMATIC SEGMENTATION
# =====================================================================

std_change = df["Rate Change"].std()
event_threshold = 6 * std_change
print("\nAutomatic Event Threshold =", round(event_threshold, 2))

df["Event"] = abs(df["Rate Change"]) > event_threshold

df["Segment"] = 0
segment = 0
counter = 0
for i in range(1, len(df)):
    counter += 1
    if df.loc[i, "Event"] and counter >= MIN_SEGMENT_LEN:
        segment += 1
        counter = 0
    df.loc[i, "Segment"] = segment

counts = df["Segment"].value_counts()
for seg in counts.index:
    if counts[seg] < MIN_SEGMENT_MERGE and seg != 0:
        df.loc[df["Segment"] == seg, "Segment"] = seg - 1

unique_seg = sorted(df["Segment"].unique())
mapping = {old: new for new, old in enumerate(unique_seg)}
df["Segment"] = df["Segment"].map(mapping)

print("\nJumlah data per segmen:")
print(df["Segment"].value_counts().sort_index())
print("\nRentang tanggal per segmen:")
print(df.groupby("Segment")["Date"].agg(["min", "max"]))


# =====================================================================
# 7. QUALITY / TREND HELPER FUNCTIONS
# =====================================================================

def monotonicity(q):
    diff = np.diff(q)
    return np.sum(diff <= 0) / len(diff)


def noise_score(q):
    mean_q = np.mean(q)
    if mean_q <= 1e-9:
        return np.inf
    return np.std(q) / mean_q


def trend_score(t, q):
    slope, intercept, r, p, std = linregress(t, q)
    return slope


def shutin_fraction(q, limit=ECONOMIC_LIMIT):
    return np.sum(q < limit) / len(q)


def stability(q):
    return np.mean(np.abs(np.diff(q)))


# =====================================================================
# 8. PILIH DATA "TAIL" (REGIME TERKINI) SECARA STRUKTURAL
# =====================================================================
# Prinsip: forecast HARUS mencerminkan kondisi sumur SEKARANG, bukan
# periode decline lama yang kebetulan kurvanya paling mulus.
# Karena itu kita SELALU mulai dari segmen paling akhir. Kalau segmen
# tsb terlalu pendek (< MIN_SEGMENT_LEN) atau tidak menunjukkan decline
# yang valid (slope >= 0, mis. karena baru saja workover / rate naik),
# gabungkan mundur dengan segmen sebelumnya, sampai ketemu rentang data
# yang cukup panjang & valid untuk fitting.

last_segment_id = df["Segment"].max()
tail_start_seg = last_segment_id
tail_df = df[df["Segment"] >= tail_start_seg].copy()

while tail_start_seg > 0:
    valid_length = len(tail_df) >= MIN_SEGMENT_LEN
    if valid_length:
        trend = trend_score(tail_df["Time"].values, tail_df["Oil bbl/d"].values)
        if trend < 0:
            break  # sudah cukup panjang & decline valid -> pakai ini
    tail_start_seg -= 1
    tail_df = df[df["Segment"] >= tail_start_seg].copy()

print(f"\nSegmen yang dipakai untuk fitting (regime terkini): Segment {tail_start_seg} s.d. {last_segment_id}")
print("Periode :", tail_df["Date"].min().date(), "-", tail_df["Date"].max().date())
print("Jumlah titik data :", len(tail_df))

if len(tail_df) < MIN_SEGMENT_LEN:
    raise SystemExit(
        "Data tidak cukup untuk membentuk segmen decline yang valid "
        f"(butuh minimal {MIN_SEGMENT_LEN} titik data dengan trend menurun)."
    )


# =====================================================================
# 9. ARPS DECLINE FUNCTIONS (RATE & KUMULATIF) -- SELALU PAKAI WAKTU LOKAL
# =====================================================================

def exponential(t, qi, Di):
    return qi * np.exp(-Di * t)


def harmonic(t, qi, Di):
    return qi / (1 + Di * t)


def hyperbolic(t, qi, Di, b):
    return qi / ((1 + b * Di * t) ** (1 / b))


def cum_exponential(t, qi, Di):
    return (qi / Di) * (1 - np.exp(-Di * t))


def cum_harmonic(t, qi, Di):
    return (qi / Di) * np.log(1 + Di * t)


def cum_hyperbolic(t, qi, Di, b):
    if np.isclose(b, 0):
        return cum_exponential(t, qi, Di)
    if np.isclose(b, 1):
        return cum_harmonic(t, qi, Di)
    return (qi / (Di * (1 - b))) * (1 - (1 + b * Di * t) ** (1 - 1 / b))


def rate_function(t, model_name, param):
    if model_name == "Exponential":
        return exponential(t, *param)
    elif model_name == "Harmonic":
        return harmonic(t, *param)
    elif model_name == "Hyperbolic":
        return hyperbolic(t, *param)


def cum_function(t, model_name, param):
    if model_name == "Exponential":
        return cum_exponential(t, *param)
    elif model_name == "Harmonic":
        return cum_harmonic(t, *param)
    elif model_name == "Hyperbolic":
        return cum_hyperbolic(t, *param)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# =====================================================================
# 10. SLIDING WINDOW DI DALAM TAIL DATA (bias ke bagian paling akhir)
# =====================================================================

def find_best_window(segment_df, window_size=None, step=None):

    if window_size is None:
        window_size = max(MIN_SEGMENT_LEN, int(len(segment_df) * 0.6))
    if step is None:
        step = max(10, int(window_size / 10))

    best_score = -999
    best_window = None

    for start in range(0, len(segment_df) - window_size + 1, step):

        window = segment_df.iloc[start:start + window_size].copy()

        q = window["Oil bbl/d"].values
        t = window["Time"].values

        mono = monotonicity(q)
        noise = noise_score(q)
        trend = trend_score(t, q)
        if trend >= 0:
            continue

        shutin = shutin_fraction(q)
        stable = stability(q)

        # Window yang lebih dekat ke ujung akhir tail_df (lebih baru)
        # sedikit diprioritaskan.
        recency_bonus = (start + window_size) / len(segment_df)

        score = (
            mono * 30
            + max(0, 1 - noise) * 20
            + max(0, 1 - shutin) * 15
            + (1 / (1 + stable)) * 15
            + max(0, -trend / 5) * 10
            + recency_bonus * 10
        )

        if score > best_score:
            best_score = score
            best_window = window.copy()

    return best_window, best_score


window_size = min(180, int(len(tail_df) * 0.7))
best_window, best_score = find_best_window(tail_df, window_size=window_size, step=15)

if best_window is None:
    # Fallback: kalau sliding window tidak menemukan sub-window valid
    # (jarang terjadi karena tail_df sendiri sudah lolos syarat trend < 0),
    # pakai seluruh tail_df apa adanya.
    print("\nPERINGATAN: sliding window tidak menemukan sub-window optimal, "
          "menggunakan seluruh rentang tail data.")
    best_window = tail_df.copy()
    best_score = np.nan


# =====================================================================
# 11. FIT ALL ARPS MODELS -- WAKTU LOKAL (t=0 di awal window)
# =====================================================================
# INI BAGIAN BUG FIX UTAMA: sebelumnya di sini dipakai "Time" absolut
# (hari sejak awal dataset), sehingga qi hasil fit tidak sama dengan
# rate di awal window. Sekarang t digeser supaya t=0 = titik pertama
# window -> qi benar-benar merepresentasikan rate di awal window, dan
# forecast (yang juga mulai dari t_local=0 di titik itu) jadi konsisten.

t_abs = best_window["Time"].values
t0 = t_abs[0]
t = t_abs - t0                      # <-- waktu LOKAL, dipakai fitting
q = best_window["Oil bbl/d"].values


def fit_all_models(t, q):

    result = {}

    qmax = np.max(q)
    qi0 = q[0]

    qi_lower = max(0.8 * qmax, q[0])
    qi_upper = 1.2 * qmax
    qi0 = min(max(qi0, qi_lower), qi_upper)

    Di0 = 0.001
    Di_lower = 1e-6
    Di_upper = 0.02

    try:
        popt, _ = curve_fit(
            exponential, t, q, p0=[qi0, Di0],
            bounds=([qi_lower, Di_lower], [qi_upper, Di_upper]), maxfev=10000
        )
        qfit = exponential(t, *popt)
        result["Exponential"] = {"RMSE": rmse(q, qfit), "R2": r2_score(q, qfit), "Parameter": popt}
    except Exception:
        result["Exponential"] = {"RMSE": np.inf, "R2": -999, "Parameter": None}

    try:
        popt, _ = curve_fit(
            harmonic, t, q, p0=[qi0, Di0],
            bounds=([qi_lower, Di_lower], [qi_upper, Di_upper]), maxfev=10000
        )
        qfit = harmonic(t, *popt)
        result["Harmonic"] = {"RMSE": rmse(q, qfit), "R2": r2_score(q, qfit), "Parameter": popt}
    except Exception:
        result["Harmonic"] = {"RMSE": np.inf, "R2": -999, "Parameter": None}

    try:
        popt, _ = curve_fit(
            hyperbolic, t, q, p0=[qi0, Di0, 0.5],
            bounds=([qi_lower, Di_lower, 1e-6], [qi_upper, Di_upper, 1.5]), maxfev=20000
        )
        qfit = hyperbolic(t, *popt)
        result["Hyperbolic"] = {"RMSE": rmse(q, qfit), "R2": r2_score(q, qfit), "Parameter": popt}
    except Exception:
        result["Hyperbolic"] = {"RMSE": np.inf, "R2": -999, "Parameter": None}

    return result


models = fit_all_models(t, q)
best_model = min(models, key=lambda x: models[x]["RMSE"])
best_param = models[best_model]["Parameter"]

if best_param is None:
    raise SystemExit("Semua model Arps gagal di-fit pada tail data ini.")

print("\n" + "=" * 50)
print("MODEL DECLINE TERPILIH (dari regime/tail terkini)")
print("=" * 50)
print("Periode window :", best_window["Date"].iloc[0].date(), "-", best_window["Date"].iloc[-1].date())
print("Model          :", best_model)
print("RMSE           :", round(models[best_model]["RMSE"], 3))
print("R2             :", round(models[best_model]["R2"], 4))
print("qi (bbl/d)     :", round(best_param[0], 2))
print("Di (1/hari)    :", round(best_param[1], 6))
if best_model == "Hyperbolic":
    print("b              :", round(best_param[2], 4))

# ---------- Validasi konsistensi (BARU) ----------
# Bandingkan prediksi model dengan data aktual, supaya penyimpangan
# langsung kelihatan di log, bukan baru ketahuan dari plot.
pred_start = rate_function(0, best_model, best_param)
pred_end = rate_function(t[-1], best_model, best_param)
t_to_last_actual = (last_date - best_window["Date"].iloc[0]).days
pred_at_last_actual = rate_function(t_to_last_actual, best_model, best_param)

print("\nValidasi konsistensi model vs data aktual:")
print(f"  Awal window   : prediksi = {pred_start:8.1f} bbl/d | aktual = {q[0]:8.1f} bbl/d")
print(f"  Akhir window  : prediksi = {pred_end:8.1f} bbl/d | aktual = {q[-1]:8.1f} bbl/d")
print(f"  Tgl data terakhir ({last_date.date()}): prediksi = {pred_at_last_actual:8.1f} bbl/d "
      f"| aktual = {last_actual_rate:8.1f} bbl/d")

deviation_pct = abs(pred_at_last_actual - last_actual_rate) / max(last_actual_rate, 1e-6) * 100
if deviation_pct > 30:
    print(f"  PERINGATAN: prediksi model meleset {deviation_pct:.0f}% dari rate aktual "
          "terakhir. Trend decline mungkin sudah berubah setelah periode window ini -- "
          "pertimbangkan window/segmen yang lebih baru atau periksa data secara manual.")


# =====================================================================
# 12. FORECAST SAMPAI ECONOMIC LIMIT (waktu lokal, konsisten dgn fitting)
# =====================================================================

t_local_max = t[-1]   # panjang window dalam waktu lokal


def rate_minus_limit(t_local):
    return rate_function(t_local, best_model, best_param) - ECONOMIC_LIMIT


if rate_function(MAX_FORECAST_DAYS, best_model, best_param) > ECONOMIC_LIMIT:
    t_econ_limit = MAX_FORECAST_DAYS
    print(f"\nCatatan: rate belum mencapai economic limit dalam "
          f"{MAX_FORECAST_DAYS} hari, forecast dipotong di batas atas.")
else:
    t_econ_limit = brentq(rate_minus_limit, t_local_max, MAX_FORECAST_DAYS)

forecast_t_local = np.linspace(0, t_econ_limit, 500)
forecast_q = rate_function(forecast_t_local, best_model, best_param)
forecast_date = best_window["Date"].iloc[0] + pd.to_timedelta(forecast_t_local, unit="D")

Np_at_econ_limit = cum_function(t_econ_limit, best_model, best_param)
EUR = Np_at_econ_limit

remaining_days = t_econ_limit - t_local_max
remaining_years = remaining_days / 365

print("\n" + "=" * 50)
print("FORECAST & EUR")
print("=" * 50)
print("Economic limit          :", ECONOMIC_LIMIT, "bbl/d")
print("Tanggal capai limit     :", forecast_date[-1].date())
print("Sisa umur produksi      : {:.1f} tahun ({:.0f} hari)".format(remaining_years, remaining_days))
print("EUR (sejak awal window) : {:,.0f} bbl".format(EUR))


# =====================================================================
# 13. PLOT
# =====================================================================

plt.figure(figsize=(14, 6))
plt.plot(df["Date"], df["Oil bbl/d"], color="lightgray", label="Actual Production (all)")
plt.plot(best_window["Date"], q, color="blue", linewidth=2, label="Data Window (fit source)")
plt.plot(best_window["Date"], rate_function(t, best_model, best_param),
         color="black", linestyle="--", linewidth=2, label=f"{best_model} Fit")
plt.plot(forecast_date, forecast_q, color="red", linewidth=2, label="Forecast")
plt.axhline(ECONOMIC_LIMIT, color="green", linestyle=":",
            label=f"Economic Limit ({ECONOMIC_LIMIT} bbl/d)")
plt.scatter([last_date], [last_actual_rate], color="black", zorder=5,
            label="Last Actual Data Point")

plt.title(f"Decline Curve Analysis (Segment {tail_start_seg}-{last_segment_id}, {best_model})")
plt.xlabel("Date")
plt.ylabel("Oil Rate (bbl/d)")
plt.legend()
plt.grid(True)
plt.show()


# =====================================================================
# 14. EXPORT KE EXCEL
# =====================================================================

summary = pd.DataFrame([{
    "Segment Range Used": f"{tail_start_seg}-{last_segment_id}",
    "Model": best_model,
    "Window Start": best_window["Date"].iloc[0],
    "Window End": best_window["Date"].iloc[-1],
    "qi (bbl/d)": best_param[0],
    "Di (1/hari)": best_param[1],
    "b": best_param[2] if best_model == "Hyperbolic" else None,
    "RMSE": models[best_model]["RMSE"],
    "R2": models[best_model]["R2"],
    "Economic Limit (bbl/d)": ECONOMIC_LIMIT,
    "Last Actual Date": last_date,
    "Last Actual Rate (bbl/d)": last_actual_rate,
    "Predicted Rate at Last Actual Date (bbl/d)": pred_at_last_actual,
    "Deviation vs Last Actual (%)": deviation_pct,
    "Forecast End Date": forecast_date[-1],
    "Remaining Years": remaining_years,
    "EUR (bbl)": EUR,
}])

try:
    with pd.ExcelWriter(OUTPUT_PATH) as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame({
            "Date": best_window["Date"].values,
            "Time_local": t,
            "Oil_actual": q,
            "Oil_fit": rate_function(t, best_model, best_param),
        }).to_excel(writer, sheet_name="Fit Window Detail", index=False)
        pd.DataFrame({
            "Date": forecast_date,
            "Oil_forecast": forecast_q,
        }).to_excel(writer, sheet_name="Forecast", index=False)

    print("\nHasil disimpan ke:", OUTPUT_PATH)

except Exception as e:
    print("\nGagal menyimpan file Excel:", e)
