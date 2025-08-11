
# 🚗 Prediksi Harga Mobil Bekas dengan XGBoost

Proyek ini bertujuan untuk membangun model Machine Learning berbasis **XGBoost** untuk memprediksi harga mobil bekas. Model ini dibangun dengan dua skenario pemodelan terpisah berdasarkan segmentasi harga: **Harga Rendah** dan **Harga Tinggi**.  

---

## 📂 Dataset
Dataset berisi informasi spesifikasi mobil bekas yang mencakup:
- **Fitur Numerik**: Tahun pembuatan, jarak tempuh, kapasitas mesin, dll.
- **Fitur Kategorik**: Merek, model, jenis bahan bakar, tipe transmisi, lokasi penjualan, dll.
- **Target**: Harga mobil bekas.

---

## 🔍 Analisis Data
1. **Hubungan Fitur Numerik dengan Target**  
   - Beberapa fitur numerik seperti tahun produksi dan jarak tempuh menunjukkan korelasi yang jelas dengan harga.
   - Terdapat pola linier menurun untuk jarak tempuh, dan pola meningkat untuk tahun produksi.
2. **Hubungan Fitur Kategorik dengan Target**  
   - Kategori tertentu (misalnya merek premium atau tipe mesin besar) memiliki harga rata-rata jauh lebih tinggi.
   - Beberapa kategori tumpang tindih dalam distribusi harga, sehingga memerlukan encoding yang tepat.

---

## ⚙️ Pemrosesan Data
Langkah-langkah utama:
1. **Pembersihan Data**: Menghapus duplikasi, menangani missing value.
2. **Feature Engineering**:  
   - Encoding fitur kategorik (One-Hot / Target Encoding).  
   - Transformasi log pada target untuk segmentasi model.
3. **Pembagian Dataset**: Train-test split sesuai segmen harga.
4. **Standarisasi Fitur** (jika diperlukan untuk model tambahan).

---

## 🤖 Pemodelan
Model yang digunakan: **XGBoost Regressor**  
Hyperparameter tuning dilakukan menggunakan **Grid Search / Random Search** untuk mendapatkan konfigurasi terbaik di setiap segmen harga.

---

## 📊 Performa Model
### **Model Harga Rendah (Setelah Tuning)**
- **R² Score**: 0.6755  
- **MSE**: 38,872,884.00  
- **RMSE**: 6,234.81  
- **MAE**: 4,504.43  
- **MAPE**: 17.67%

### **Model Harga Tinggi (Setelah Tuning)**
- **R² Score**: 0.8305  
- **MSE**: 974,854,784.00  
- **RMSE**: 31,222.66  
- **MAE**: 16,683.27  
- **MAPE**: 13.69%

---

## 📌 Kesimpulan
- Model pada segmen **harga tinggi** menunjukkan performa yang lebih baik (R²: 0.83) dibandingkan segmen harga rendah (R²: 0.67).  
- Perbedaan ini dapat disebabkan oleh variasi harga yang lebih terstruktur di pasar mobil premium dibandingkan mobil ekonomis.  
- Beberapa fitur kategorik dan numerik terbukti sangat mempengaruhi harga, sehingga dapat digunakan untuk peningkatan model lebih lanjut.

---

## 🚀 Penggunaan
1. Clone repository:
   ''' bash
   git clone https://github.com/Haki-X/Streamlit_Porto1
   ```bash
   git clone https://github.com/username/nama-repo.git
