# 📘 Laporan Teknis Proyek
## Fintech Credit Risk System – Sistem Analisis Risiko Kredit Digital Berbasis Machine Learning

---

## 1. Pendahuluan
Perkembangan teknologi finansial (*financial technology / fintech*) mendorong lembaga keuangan untuk mengadopsi sistem analisis risiko kredit yang lebih cepat, akurat, dan berbasis data. Salah satu permasalahan utama dalam proses pemberian pinjaman adalah risiko gagal bayar (*default risk*) yang dapat berdampak langsung pada stabilitas keuangan lembaga.

Proyek **Fintech Credit Risk System** dikembangkan sebagai sebuah sistem analisis risiko kredit digital yang memanfaatkan algoritma *Machine Learning* untuk memprediksi kemungkinan gagal bayar berdasarkan karakteristik peminjam. Sistem ini mengintegrasikan **backend FastAPI**, **model Random Forest**, dan **frontend web interaktif**.

---

## 2. Tujuan Proyek
Tujuan pengembangan sistem ini adalah:
1. Membangun model prediksi risiko kredit berbasis *Machine Learning*.
2. Mengklasifikasikan tingkat risiko kredit menjadi **LOW**, **MEDIUM**, dan **HIGH**.
3. Menyediakan REST API untuk prediksi risiko kredit secara real-time.
4. Mengembangkan antarmuka web yang informatif dan mudah digunakan.
5. Menyediakan dokumentasi teknis yang terstruktur.

---

## 3. Ruang Lingkup Sistem
Ruang lingkup sistem meliputi:
- Pengolahan dataset kredit.
- Pelatihan dan evaluasi model *Machine Learning*.
- Penyediaan API prediksi risiko kredit.
- Visualisasi hasil prediksi melalui antarmuka web.
- Penyajian aset statis (gambar dan video) melalui FastAPI.

---

## 4. Arsitektur Sistem
Sistem menggunakan arsitektur **client–server** yang terdiri dari:
- **Frontend**: HTML, Tailwind CSS, JavaScript.
- **Backend**: FastAPI.
- **Model**: Random Forest Classifier (Scikit-learn).

---

## 5. Dataset dan Fitur
### 5.1 Dataset
Dataset yang digunakan adalah *loan_default.csv* yang berisi data historis peminjam.

### 5.2 Fitur Input
| Fitur | Deskripsi |
|------|----------|
| age | Usia peminjam |
| income | Pendapatan peminjam |
| loanamount | Jumlah pinjaman |
| creditscore | Skor kredit |

### 5.3 Target
- **default** → Status gagal bayar (0 = tidak, 1 = ya)

---

## 6. Proses Pelatihan Model
1. Dataset dibaca dan dinormalisasi.
2. Data dibagi menjadi data latih dan data uji (80:20).
3. Model Random Forest dilatih menggunakan data latih.
4. Evaluasi dilakukan menggunakan *accuracy score*.
5. Model disimpan dalam format `.pkl`.

---

## 7. Mekanisme Prediksi Risiko
Probabilitas gagal bayar dihitung oleh model dan diklasifikasikan sebagai:
| Probabilitas | Level Risiko |
|-------------|-------------|
| < 0.20 | LOW |
| 0.20 – < 0.30 | MEDIUM |
| ≥ 0.30 | HIGH |

---

## 8. Implementasi API
### Endpoint Prediksi
`POST /predict`

### Contoh Request
```json
{
  "Age": 35,
  "Income": 7000000,
  "LoanAmount": 20000000,
  "CreditScore": 680
}
```

### Contoh Response
```json
{
  "default_prediction": 0,
  "default_probability": 0.1842,
  "risk_level": "LOW",
  "threshold_used": 0.3
}
```

---

## 9. Struktur Direktori
```
project/
├── main.py
├── train.py
├── loan_default.csv
├── loan_default_model.pkl
├── assets/
│   ├── bank.jpeg
│   └── bank.mp4
├── templates/
│   └── index.html
├── requirements.txt
└── README.md
```

---

## 10. Cara Menjalankan Sistem
### Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Training Model
```bash
python train.py
```

### Menjalankan API
```bash
uvicorn main:app --reload
```

Akses aplikasi:
```
http://127.0.0.1:8000
```

---

## 11. Kesimpulan
Sistem ini berhasil mengimplementasikan analisis risiko kredit berbasis *Machine Learning* yang terintegrasi dengan API dan antarmuka web, sehingga dapat digunakan sebagai dasar pengembangan aplikasi fintech yang lebih kompleks.

---

## 12. Referensi
- FastAPI: https://fastapi.tiangolo.com/
- Scikit-learn: https://scikit-learn.org/
- Referensi kode: https://www.scirp.org/journal/paperinformation?paperid=132019
