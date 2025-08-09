# =============================================================================
# APLIKASI WEB PREDIKSI HARGA MOBIL DENGAN STREAMLIT
# File: app.py (VERSI PERBAIKAN)
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# Konfigurasi Awal & Memuat Model
# =============================================================================

@st.cache_resource
def load_models():
    """Memuat model spesialis dari file .pkl"""
    try:
        model_low = joblib.load('model_low_price_tuned2.pkl') # Sesuaikan nama file jika berbeda
        model_high = joblib.load('model_high_price_tuned2.pkl') # Sesuaikan nama file jika berbeda
        return model_low, model_high
    except FileNotFoundError:
        st.error("Error: Pastikan file model .pkl ada di folder yang sama dengan app.py.")
        return None, None

model_low, model_high = load_models()
PRICE_THRESHOLD = 49000.00 

# =============================================================================
# FUNGSI FEATURE ENGINEERING (Disalin dari Notebook Training)
# =============================================================================
def create_enhanced_features(df):
    """
    Fungsi ini WAJIB sama persis dengan yang ada di notebook training
    untuk memastikan konsistensi fitur.
    """
    df_eng = df.copy()
    current_year = pd.to_datetime('today').year
    
    df_eng['Car_Age'] = current_year - df_eng['Year']
    df_eng['Mileage_per_Year'] = df_eng['Mileage'] / (df_eng['Car_Age'] + 1)
    
    luxury_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Lexus', 'Infiniti', 'Cadillac', 'Jaguar', 'Land Rover', 'Porsche', 'GMC']
    premium_brands = ['Toyota', 'Honda', 'Nissan', 'Ford', 'Chevrolet', 'Hyundai', 'Kia', 'Mazda', 'Jeep', 'Dodge']
    
    def get_brand_tier(make):
        if make in luxury_brands: return 'luxury'
        elif make in premium_brands: return 'premium'
        else: return 'economy'
    df_eng['Brand_Tier'] = df_eng['Make'].apply(get_brand_tier).astype(str)
    
    luxury_types = ['G', 'LX', 'Range Rover', 'GLC', 'CLA', 'A8', 'X', 'S', 'LS', 'A', 'E', 'The 7', 'RX', 'C', 'Land Cruiser', 'Wrangler', 'GS', 'Escalade', 'IS', 'ES', 'Patrol', 'F150', 'Prado', 'G80', 'The 5', 'Camaro', 'FJ', 'Sierra', 'Other', 'Land Cruiser Pickup', 'CX5', 'Tahoe', 'Ciocca', 'Avalon', 'Mustang', 'CLS', 'Charger', 'CX9', 'Yukon', 'Furniture', 'S300', 'A6', 'Grand Cherokee', 'Silverado']
    premium_types = ['Land Cruiser 70', 'Copper', '500', 'Prestige', 'Rav4', 'Hilux', 'H6', 'Kona', 'Cadenza', 'C300', 'Challenger', 'Explorer', 'Impala', 'Suburban', 'Taurus', 'Accord', 'D-MAX', 'SEL', 'VTC', 'Expedition', 'Odyssey', 'Azera', 'The 3', 'CT-S', 'Pajero', 'Durango', '300', 'Tucson', 'Senta fe', 'Hiace', 'CX3', 'Sportage', 'Creta', 'Caprice', 'CS75', 'X-Trail', '6', 'Innova', 'Sonata', 'ML', 'Civic', 'Camry', 'Altima']

    def get_type_tier(car_type):
        if car_type in luxury_types: return 'luxury_type'
        elif car_type in premium_types: return 'premium_type'
        else: return 'economy_type'
    df_eng['Type_Tier'] = df_eng['Type'].apply(get_type_tier).astype(str)
    
    def categorize_age(age):
        if age <= 2: return 'new'
        elif age <= 5: return 'recent'
        elif age <= 10: return 'mature'
        else: return 'old'
    df_eng['Age_Category'] = df_eng['Car_Age'].apply(categorize_age).astype(str)
    
    df_eng['Mileage_Category'] = pd.cut(df_eng['Mileage'], bins=[0, 50000, 100000, 200000, float('inf')], labels=['low', 'medium', 'high', 'very_high']).astype(str)
    df_eng['Engine_Category'] = pd.cut(df_eng['Engine_Size'], bins=[0, 1.5, 2.5, 3.5, float('inf')], labels=['small', 'medium', 'large', 'very_large']).astype(str)
    df_eng['Age_Mileage_Interaction'] = df_eng['Car_Age'] * df_eng['Mileage'] / 1000
    
    # Fitur interaksi yang diminta oleh model
    df_eng['Make_x_Type'] = df_eng['Make'] + '_' + df_eng['Type']
    df_eng['Make_x_Age'] = df_eng['Make'] + '_age_' + df_eng['Car_Age'].astype(str)

    # Hapus kolom asli yang sudah tidak diperlukan lagi oleh model
    # Hapus 'Year', bukan 'Price' atau 'Price_log' karena ini bukan input untuk model
    df_eng.drop(columns=['Year'], inplace=True, errors='ignore')
    
    return df_eng

# =============================================================================
# Fungsi Prediksi Utama (Sama seperti sebelumnya)
# =============================================================================
def predict_price(car_features_df):
    initial_pred_log = model_high.predict(car_features_df)
    initial_pred_original = np.expm1(initial_pred_log)
    
    model_used = ""
    if initial_pred_original[0] < PRICE_THRESHOLD:
        model_used = "Spesialis Harga Rendah"
        final_pred_log = model_low.predict(car_features_df)
    else:
        model_used = "Spesialis Harga Tinggi"
        final_pred_log = initial_pred_log
        
    final_price = np.expm1(final_pred_log)
    return final_price[0], model_used

# =============================================================================
# Tampilan Antarmuka (User Interface) Aplikasi Web
# =============================================================================

st.set_page_config(page_title="Prediksi Harga Mobil Bekas", layout="wide")
st.title("🚗 Estimator Harga Mobil Bekas")
st.write("Aplikasi ini menggunakan Machine Learning untuk memprediksi harga wajar mobil bekas di Arab Saudi.")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.header("Fitur Utama Mobil")
    # Contoh daftar, idealnya ini harus diambil dari data training
    list_of_makes = ['Toyota', 'Hyundai', 'Ford', 'Chevrolet', 'Nissan', 'Kia', 'Mercedes', 'GMC', 'Lexus', 'Mazda'] 
    list_of_types = ['Corolla', 'Camry', 'Yaris', 'Elantra', 'Accent', 'Land Cruiser', 'Hilux', 'Sonata', 'Accord']
    make = st.selectbox("Merek (Make)", options=sorted(list_of_makes))
    car_type = st.selectbox("Tipe (Type)", options=sorted(list_of_types))
    year = st.number_input("Tahun Pembuatan (Year)", min_value=1960, max_value=2025, value=2018, step=1)
    mileage = st.number_input("Jarak Tempuh (Mileage)", min_value=100, max_value=1000000, value=120000, step=1000)

with col2:
    st.header("Spesifikasi Tambahan")
    gear_type = st.selectbox("Tipe Gigi (Gear Type)", options=['Automatic', 'Manual'])
    origin = st.selectbox("Asal (Origin)", options=['Saudi', 'Gulf Arabic', 'Other', 'Unknown'])
    options = st.selectbox("Opsi (Options)", options=['Full', 'Semi Full', 'Standard'])
    engine_size = st.slider("Ukuran Mesin (Engine Size)", min_value=1.0, max_value=9.0, value=2.0, step=0.1)

if st.button("Prediksi Harga", type="primary", use_container_width=True):
    if model_low is not None and model_high is not None:
        # 1. Kumpulkan input mentah
        input_data = {
            'Make': make, 'Type': car_type, 'Gear_Type': gear_type,
            'Origin': origin, 'Options': options, 'Mileage': mileage,
            'Engine_Size': engine_size, 'Year': year
        }
        input_df_raw = pd.DataFrame([input_data])
        
        # 2. PERBAIKAN: Panggil fungsi create_enhanced_features
        # untuk membuat SEMUA fitur yang dibutuhkan model
        input_df_engineered = create_enhanced_features(input_df_raw)
        
        # 3. Panggil fungsi prediksi dengan DataFrame yang sudah di-engineer
        with st.spinner('Model sedang menganalisis...'):
            predicted_price, model_name = predict_price(input_df_engineered)
        
        # 4. Tampilkan hasil
        st.markdown("---")
        st.header("Hasil Prediksi")
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric(label="Estimasi Harga Wajar", value=f"{predicted_price:,.0f} SAR")
        with col_res2:
            st.info(f"Model yang Digunakan: **{model_name}**")
        st.success("Prediksi berhasil dibuat!")
    else:
        st.error("Model tidak dapat dimuat. Proses tidak dapat dilanjutkan.")