# Memuat library yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Memuat Data dengan pandas
data_path = 'prakiraan_cuaca_dummy.csv'
data = pd.read_csv(data_path)

# Menampilkan lima baris pertama
print("Lima Baris Pertama Dataset:")
print(data.head())

# Memeriksa Data
print("\nInformasi Struktur Data:")
print(data.info())

print("\nJumlah Missing Value:")
print(data.isnull().sum())

# Mengisi missing value dengan mean untuk kolom numerik
data['Suhu (Celsius)'] = data['Suhu (Celsius)'].fillna(data['Suhu (Celsius)'].mean())
data['Tebal Awan (meter)'] = data['Tebal Awan (meter)'].fillna(data['Tebal Awan (meter)'].mean())

print("\nJumlah Missing Value per Kolom Setelah Imputasi:")
print(data.isnull().sum())

# Mengubah tipe data kolom 'Prakiraan Cuaca' menjadi kategori
data['Prakiraan Cuaca'] = data['Prakiraan Cuaca'].astype('category')

# **Visualisasi Distribusi Sebelum Dummy Encoding**
plt.figure(figsize=(6, 4))
sns.countplot(x='Prakiraan Cuaca', data=data, palette='viridis')
plt.title('Distribusi Prakiraan Cuaca (Sebelum Dummy Encoding)')
plt.xlabel('Prakiraan Cuaca')
plt.ylabel('Frekuensi')
plt.savefig('distribusi_prakiraan_cuaca_sebelum.png')
plt.show()

# Mengubah data kategorikal menjadi dummy variables
data = pd.get_dummies(data, columns=['Prakiraan Cuaca'])

# Memeriksa struktur data setelah dummy encoding
print("\nInformasi Struktur Data Setelah Dummy Encoding:")
print(data.info())

# Mengidentifikasi Outlier
def deteksi_outlier(df, kolom):
    Q1 = df[kolom].quantile(0.25)
    Q3 = df[kolom].quantile(0.75)
    IQR = Q3 - Q1
    batas_bawah = Q1 - 1.5 * IQR
    batas_atas = Q3 + 1.5 * IQR
    outliers = df[(df[kolom] < batas_bawah) | (df[kolom] > batas_atas)]
    return outliers

info_outlier = {}
kolom_numerik = ['Suhu (Celsius)', 'Kelembaban (%)', 'Kecepatan Angin (km/jam)', 
                 'Tebal Awan (meter)', 'Tekanan Atmosfer (hPa)']
for kolom in kolom_numerik:
    outliers = deteksi_outlier(data, kolom)
    info_outlier[kolom] = len(outliers)

print("\nJumlah Outlier per Kolom:")
print(info_outlier)

# Menghapus duplikasi
data = data.drop_duplicates()
print("\nJumlah Baris Duplikat Setelah Dihapus:")
print(data.duplicated().sum())

# Normalisasi kolom numerik
numerical_columns = ['Suhu (Celsius)', 'Kelembaban (%)', 'Kecepatan Angin (km/jam)', 
                     'Tebal Awan (meter)', 'Tekanan Atmosfer (hPa)']
scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

print("\nLima Baris Pertama Setelah Normalisasi:")
print(data.head())

# Menyimpan dataset yang diperbarui
output_path = 'hasil_prakiraan_cuaca.csv'
data.to_csv(output_path, index=False)
print(f"\nDataset yang diperbarui telah disimpan ke file: {output_path}")

# **Visualisasi Distribusi Setelah Dummy Encoding**
# Hitung distribusi berdasarkan kolom dummy
kategori_dummy = [col for col in data.columns if 'Prakiraan Cuaca_' in col]
distribusi_cuaca = data[kategori_dummy].sum()

plt.figure(figsize=(6, 4))
distribusi_cuaca.plot(kind='bar', color='skyblue')
plt.title('Distribusi Prakiraan Cuaca (Setelah Dummy Encoding)')
plt.xlabel('Kategori Prakiraan Cuaca')
plt.ylabel('Frekuensi')
plt.savefig('distribusi_prakiraan_cuaca_setelah.png')
plt.show()

# **Boxplot untuk Kolom Numerik**
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(data=data, y=col, palette='viridis')
    plt.title(f'Distribusi {col}')
    plt.ylabel(col)
plt.tight_layout()
plt.savefig('boxplot_numerical.png')
plt.show()

# **Heatmap Korelasi**
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap Korelasi Antar Variabel')
plt.savefig('heatmap_korelasi.png')
plt.show()
