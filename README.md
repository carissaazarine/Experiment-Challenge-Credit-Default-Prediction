[README_credit_default.md](https://github.com/user-attachments/files/34216717/README_credit_default.md)
# Experiment Challenge: Credit Default Prediction

Proyek mata kuliah Prediksi Modern dan Machine Learning. Tujuannya memprediksi apakah nasabah kartu kredit akan gagal bayar pada bulan berikutnya, melalui serangkaian eksperimen terstruktur yang membandingkan pra-pemrosesan, model, dan penyetelan hyperparameter.

## Dataset

**UCI Credit Card (Default of Credit Card Clients)**: 30.000 nasabah dan 25 variabel (limit kredit, jenis kelamin, pendidikan, status pernikahan, usia, riwayat pembayaran `PAY_x`, tagihan, dan pembayaran). Target: `default.payment.next.month`.

Temuan awal:
- Tidak ada missing value dan tidak ada data duplikat.
- Nilai 0 pada `EDUCATION` dan `MARRIAGE` tidak ada dalam dokumentasi dataset, sehingga direkode ke kategori *others*.
- Kelas target tidak seimbang: 77,88% tidak default dan 22,12% default. Karena itu **F1-score** dipakai sebagai metrik utama, bukan akurasi.
- `ID` dikeluarkan dari prediktor, sehingga tersisa 23 prediktor. Data dibagi 80:20 secara stratified (24.000 data latih, 6.000 data uji, `random_state=42`).

## Eksperimen

| # | Eksperimen | Model | CV F1 | Test F1 |
|---|---|---|---|---|
| 1 | Baseline | Logistic Regression | 0,3704 | 0,3567 |
| 2 | Pra-pemrosesan (scaling) | Logistic Regression + StandardScaler | 0,3660 | 0,3586 |
| 3 | Perbandingan model | Random Forest | 0,4696 | 0,4589 |
| 4 | Hyperparameter tuning | **Random Forest + GridSearchCV** | **0,5444** | **0,5426** |
| 5 | Model lanjutan | Gradient Boosting | 0,4780 | 0,4675 |

Pada Eksperimen 3, Random Forest mengungguli Logistic Regression (CV F1 0,3660) dan Decision Tree (CV F1 0,4078).

## Model Terbaik

**Random Forest hasil tuning** (GridSearchCV, 5-fold, metrik F1):

- Parameter terbaik: `class_weight='balanced'`, `max_depth=10`, `min_samples_leaf=2`, `min_samples_split=2`, `n_estimators=100`

| Metrik (data uji) | Skor |
|---|---|
| Accuracy | 0,7887 |
| Precision | 0,5204 |
| Recall | 0,5667 |
| F1-score | 0,5426 |

## Temuan Utama

- **Akurasi bisa menyesatkan.** Logistic Regression baseline mencapai akurasi 0,81, tetapi recall hanya 0,24, sehingga lebih dari 75% nasabah default tidak terdeteksi.
- **Penanganan ketidakseimbangan kelas berdampak besar.** Pengaturan `class_weight='balanced'` menaikkan recall dari 0,36 menjadi 0,57, dengan kompromi akurasi dan precision yang sedikit turun.
- **Scaling tidak banyak membantu** Logistic Regression: Test F1 naik tipis, tetapi CV F1 justru turun.
- Gradient Boosting memiliki akurasi (0,8185) dan precision (0,6657) lebih tinggi, tetapi recall dan F1-score lebih rendah dari Random Forest hasil tuning.

## Teknologi

Python, pandas, NumPy, matplotlib, scikit-learn (Pipeline, StandardScaler, GridSearchCV, cross_val_score). Dikerjakan di Google Colab.
