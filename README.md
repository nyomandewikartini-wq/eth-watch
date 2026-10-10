# Ethereum Price Forecasting & Market Monitoring

Aplikasi dashboard berbasis Streamlit untuk memantau harga Ethereum (ETH-USD), melakukan peramalan harga menggunakan Long Short-Term Memory (LSTM), serta mengidentifikasi pola dan anomali pasar melalui pendekatan Complex Event Processing (CEP).

## Fitur Utama

- **Price Monitoring:** Menampilkan data historis harga Ethereum.
- **Price Forecasting:** Memperkirakan harga penutupan ETH untuk satu hari ke depan menggunakan model LSTM.
- **Anomaly Detection:** Mengidentifikasi pergerakan harga yang tidak biasa berdasarkan aturan deteksi yang diterapkan.
- **Market Pattern Monitoring:** Memantau pola pergerakan harga dan hubungan harga-volume sesuai logika CEP dalam aplikasi.

## Metode

### LSTM Forecasting

Model menggunakan harga penutupan (`Close`) Ethereum dengan konfigurasi terbaik hasil tuning:

- Lookback window: 14 hari
- LSTM units: 128
- Best epoch: 12
- Forecast horizon: 1 hari
- Model selection criterion: Lowest validation RMSE
- Validation RMSE: 153.21 USD

Nilai RMSE di atas merupakan hasil validasi model, bukan jaminan kesalahan prediksi pada data baru.

### Data Source

Data harga Ethereum diperoleh dari Yahoo Finance menggunakan ticker `ETH-USD` melalui library `yfinance`.

Dashboard menggunakan data historis yang diambil ketika aplikasi dijalankan. Pengambilan data historis ini tidak dengan sendirinya berarti aplikasi menerima streaming pasar secara real-time.

## Project Structure

```text
ETH_Watch/
├── eth_app.py
├── requirements.txt
├── .python-version
├── .gitignore
├── README.md
└── models/
    ├── eth_lstm_best.keras
    ├── eth_close_scaler.joblib
    └── eth_lstm_config.json
```

## Requirements

- Python 3.11
- TensorFlow 2.16.1
- Streamlit
- NumPy
- Pandas
- scikit-learn
- Joblib
- Plotly
- yfinance

Daftar dependensi lengkap tersedia di `requirements.txt`.

## Run Locally

1. Clone repository:

   ```bash
   git clone <REPOSITORY_URL>
   cd ETH_Watch
   ```

2. Buat dan aktifkan virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows:

   ```bash
   .venv\Scripts\activate
   ```

3. Install dependensi:

   ```bash
   pip install -r requirements.txt
   ```

4. Jalankan aplikasi:

   ```bash
   streamlit run eth_app.py
   ```

## Deployment

Aplikasi dirancang untuk dapat di-deploy menggunakan Streamlit Community Cloud dengan menghubungkan repository GitHub dan memilih file utama `eth_app.py`.

Pastikan file model, scaler, dan konfigurasi tersedia pada lokasi yang diharapkan aplikasi.

## Limitations

- Hasil peramalan merupakan estimasi model, bukan jaminan harga aktual.
- Performa model dapat berubah ketika kondisi pasar berubah.
- Kualitas hasil bergantung pada ketersediaan data dan konsistensi preprocessing.
- Deteksi anomali dan pola pasar bergantung pada aturan CEP yang diterapkan.
- Dashboard ini bukan rekomendasi investasi.

## Disclaimer

Proyek ini dibuat untuk tujuan pembelajaran, analisis data, dan demonstrasi pemodelan deret waktu. Keputusan investasi tetap memerlukan analisis dan pertimbangan mandiri.