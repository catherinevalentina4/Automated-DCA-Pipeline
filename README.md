# Automated Decline Curve Analysis (DCA) untuk Sumur Minyak

Pipeline Python untuk menganalisis riwayat produksi minyak per sumur secara otomatis:
membersihkan data, mendeteksi shut-in & kemungkinan workover, mensegmentasi riwayat
produksi menjadi beberapa rezim, memilih rezim yang paling merepresentasikan kondisi
sumur *saat ini*, lalu fitting model decline Arps (Exponential / Harmonic / Hyperbolic)
untuk forecasting rate produksi hingga economic limit dan menghitung EUR (Estimated
Ultimate Recovery).

Dibuat untuk kebutuhan analisis internal pada data produksi lapangan migas, dengan
tujuan mengurangi waktu analisis manual yang biasanya dilakukan satu per satu per sumur.

> **Catatan data:** Contoh/skrip di repo ini menggunakan data sintetis untuk demonstrasi.

---

## Apa yang dilakukan pipeline ini

1. **Load & cleaning** — parsing tanggal, hapus duplikat/nilai kosong/produksi negatif.
2. **Deteksi shut-in & workover** — flag hari-hari dengan rate di bawah economic limit,
   dan lonjakan rate yang mengindikasikan kemungkinan workover.
3. **Segmentasi otomatis** — memecah riwayat produksi menjadi beberapa segmen/rezim
   berdasarkan perubahan rate yang signifikan (event detection berbasis standar deviasi).
4. **Pemilihan rezim terkini secara struktural** — alih-alih memilih segmen dengan kurva
   "paling mulus" di seluruh riwayat (yang bisa saja sudah usang / tidak lagi relevan),
   pipeline ini selalu memprioritaskan segmen paling akhir, dan hanya mundur ke segmen
   sebelumnya kalau segmen terakhir belum cukup panjang/valid untuk fitting.
5. **Sliding window search** — mencari sub-rentang data terbaik di dalam rezim terkini
   berdasarkan skor kualitas (monotonicity, noise, stabilitas, kedekatan ke waktu sekarang).
6. **Fitting 3 model Arps** (Exponential, Harmonic, Hyperbolic) — pilih model dengan RMSE
   terbaik.
7. **Forecast & EUR** — proyeksi rate sampai economic limit, hitung sisa umur produksi
   dan EUR.
8. **Validasi otomatis** — membandingkan prediksi model dengan rate aktual terakhir yang
   tersedia, dan memunculkan peringatan eksplisit kalau selisihnya besar (indikasi trend
   decline sudah berubah sejak periode fitting).

---

## Lessons Learned: Jebakan "Absolute vs Local Time" pada Curve Fitting Arps

Bagian ini yang menurut saya paling layak dibagikan, karena setelah menelusuri beberapa
tutorial/implementasi DCA otomatis lain (termasuk beberapa yang open-source di GitHub),
jebakan ini jarang dibahas eksplisit — padahal berpotensi membuat forecast **overestimate
secara sistematis**, dan efeknya makin parah semakin baru/jauh window fitting-nya dari
awal riwayat data.

### Bug-nya

Pada implementasi awal, model Arps di-fit menggunakan kolom waktu **absolut**
(jumlah hari sejak tanggal data *paling awal di seluruh riwayat sumur*, bisa dari
beberapa tahun sebelum window fitting):

```python
t = window["Time"].values          # absolut, mis. 2008–2187 (hari sejak awal dataset)
popt, _ = curve_fit(exponential, t, q, ...)
qi, Di = popt
```

Tapi saat forecasting, kurva dievaluasi memakai waktu **lokal** yang seolah-olah mulai
dari 0 di awal window:

```python
forecast_t_local = np.linspace(0, t_econ_limit, 500)   # mulai dari 0
forecast_q = exponential(forecast_t_local, qi, Di)      # qi & Di dari fit ABSOLUT
```

Karena `qi` hasil fit terikat ke referensi *t = 0 di awal dataset* (bertahun-tahun
sebelum window), memakainya kembali di *t_local = 0* menghasilkan rate yang jauh lebih
tinggi dari kondisi aktual window tersebut.

### Bukti numerik (dari data uji)

Untuk satu window fitting (durasi ±6 bulan, beberapa tahun setelah awal riwayat data):

| | Fit pakai waktu **absolut** (bug) | Fit pakai waktu **lokal** (fix) |
|---|---|---|
| qi hasil fit | 73.2 bbl/d | **48.8 bbl/d** |
| Prediksi vs rate aktual di awal window | 73.2 vs 48 (meleset **+52%**) | 48.8 vs 48 (hampir pas) |
| Sisa umur produksi hasil forecast | 29.5 tahun | **5.8 tahun** |
| EUR | 277.789 bbl | **44.191 bbl** |

Fix-nya sederhana secara kode — geser waktu window supaya *t = 0 selalu di titik
pertama window fitting*, dan pakai referensi itu secara konsisten baik untuk fitting
maupun forecasting:

```python
t_abs = window["Time"].values
t0 = t_abs[0]
t = t_abs - t0          # waktu LOKAL: t=0 di awal window
popt, _ = curve_fit(exponential, t, q, ...)
# forecast_t_local juga dihitung relatif terhadap t0 yang sama
```

### Kenapa ini penting dibagikan

Bug ini bukan salah tulis rumus Arps — rumusnya benar. Ini murni soal **konsistensi
frame referensi waktu** antara tahap fitting dan tahap penggunaan hasil fit, yang mudah
lolos review karena:
- Plot bisa tetap terlihat "masuk akal" secara visual kalau window fitting-nya kebetulan
  dekat dengan awal dataset (efek bug kecil).
- Error-nya tidak memunculkan exception apa pun — `curve_fit` tetap sukses, angkanya
  tetap "masuk akal" sekilas (qi 73 bbl/d bukan angka yang aneh untuk sumur minyak).
- Baru ketahuan kalau membandingkan qi hasil fit dengan rate aktual di titik data
  yang sama — validasi yang sering dilewatkan kalau pipeline-nya "auto-pilot" penuh.

---

## Keterbatasan yang masih terbuka

Supaya jujur soal apa yang belum selesai:

- Bobot pada quality-scoring segmen (mono/noise/shutin/stability) masih heuristik,
  belum divalidasi terhadap hasil DCA manual dari reservoir engineer.
- Pada data tail yang sangat noisy/intermiten, kualitas fit (R²) bisa rendah — model
  Arps sendiri kurang cocok untuk pola shut-in berulang jangka pendek; smoothing atau
  exclude periode shut-in dari data fitting adalah perbaikan lanjutan yang masih perlu
  dikerjakan.
- Belum ada uncertainty quantification (P10/P50/P90) pada forecast — saat ini hanya
  estimasi titik (deterministic).
- Belum dibandingkan head-to-head dengan hasil DCA manual atau software komersial.

---

## Referensi & related work

Segmentasi otomatis + fitting Arps otomatis bukan pendekatan baru di industri migas;
proyek ini adalah implementasi/adaptasi untuk kasus data produksi sumur tunggal dengan
riwayat panjang dan noisy. Beberapa referensi terkait:

- Automated DCA dengan event detection & quantile regression untuk evaluasi portofolio
  sumur skala besar (SPE, "A Fit-For-Purpose Automated Decline Curve Analysis Using
  Python and BI Dashboard").
- Implementasi open-source serupa yang memakai change-point detection untuk
  segmentasi otomatis sebelum fitting Arps.

---

## Cara pakai

```bash
pip install -r requirements.txt
python dca_single_well.py --file path/to/data.xlsx --output path/to/result.xlsx
```

Format data input: file Excel dengan kolom `Date` dan `Oil bbl/d`.

---

## Tech stack

Python · pandas · numpy · scipy (curve_fit, brentq) · scikit-learn (metrics) ·
matplotlib
