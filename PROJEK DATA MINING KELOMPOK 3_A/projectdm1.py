# Memuat library yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# 1. Memuat Data
file_path = 'file_dataset.csv'
data = pd.read_csv(file_path)

# 2. Memeriksa Data
print("Informasi Dataset:")
print(data.info())
print("\n5 Baris Pertama Data:")
print(data.head())

# 3. Mengidentifikasi Missing Values
print("\nMissing Values per Kolom:")
print(data.isnull().sum())

# 4. Mengubah Tipe Data
# Mengidentifikasi kolom dengan tipe data yang perlu diubah
for col in data.columns:
    if data[col].dtype == 'object':
        try:
            data[col] = pd.to_numeric(data[col], errors='ignore')  # Coba ubah ke numerik jika memungkinkan
        except Exception as e:
            print(f"Kolom {col} tidak dapat diubah: {e}")

# 5. Mengidentifikasi Outliers
def detect_outliers_zscore(dataframe, threshold=3):
    """Deteksi outlier menggunakan Z-score."""
    z_scores = (dataframe - dataframe.mean()) / dataframe.std()
    return (np.abs(z_scores) > threshold)

outliers = {}
for col in data.select_dtypes(include=[np.number]).columns:
    outliers[col] = detect_outliers_zscore(data[col]).sum()

print("\nJumlah Outliers per Kolom:")
print(outliers)

# 6. Menangani Outliers
# Contoh menangani outlier dengan capping
for col in data.select_dtypes(include=[np.number]).columns:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data[col] = np.where(data[col] < lower_bound, lower_bound,
                         np.where(data[col] > upper_bound, upper_bound, data[col]))

# 7. Deteksi dan Hapus Duplikasi
print("\nJumlah Duplikasi Data:")
print(data.duplicated().sum())
data = data.drop_duplicates()

# 8. Normalisasi/Standarisasi
# Normalisasi (Min-Max Scaling)
scaler = MinMaxScaler()
numerical_cols = data.select_dtypes(include=[np.number]).columns
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Menampilkan data hasil akhir
print("\nData Setelah Normalisasi/Standarisasi:")
print(data.head())

# Menyimpan dataset hasil
output_path = 'prepared_dataset.csv'
data.to_csv(output_path, index=False)
print(f"Dataset hasil persiapan disimpan di: {output_path}")

# 9. Statistik Deskriptif
print("\nStatistik Deskriptif Dataset:")
print(data.describe())

# 10. Analisis Outcome (Target)
plt.figure(figsize=(6, 4))
sns.countplot(x='Outcome', data=data, palette='viridis')
plt.title('Distribusi Outcome')
plt.xlabel('Outcome')
plt.ylabel('Frekuensi')
plt.savefig('outcome.png')
plt.show()

# 11. Distribusi Data Berdasarkan Outcome
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x='Outcome', y=col, data=data, palette='viridis')
    plt.title(f'{col} vs Outcome')
    plt.xlabel('Outcome')
    plt.ylabel(col)
plt.tight_layout()
plt.savefig('distribusiOutcome.png')
plt.show()

# Pastikan matriks korelasi telah dihitung
correlation_matrix = data.select_dtypes(include=[np.number]).corr()

# 12. Analisis Korelasi dengan Target
if 'Outcome' in correlation_matrix.columns:
    correlation_with_target = correlation_matrix['Outcome'].sort_values(ascending=False)
    print("\nKorelasi dengan Outcome:")
    print(correlation_with_target)
else:
    print("Kolom 'Outcome' tidak ditemukan dalam matriks korelasi.")

# 13. Scatterplot untuk Variabel Penting
# Memilih 3 fitur dengan korelasi tertinggi terhadap Outcome, selain Outcome itu sendiri
important_features = correlation_with_target.index[1:4]  # Abaikan 'Outcome' sebagai korelasi dengan dirinya sendiri
plt.figure(figsize=(15, 5))
for i, col in enumerate(important_features, 1):
    plt.subplot(1, 3, i)
    sns.scatterplot(x=col, y='Outcome', data=data, hue='Outcome', palette='viridis')
    plt.title(f'{col} vs Outcome')
    plt.xlabel(col)
    plt.ylabel('Outcome')
plt.tight_layout()
plt.savefig('scatterplot.png')
plt.show()

# 14. Heatmap untuk Subset Korelasi
plt.figure(figsize=(10, 8))
subset_corr = correlation_matrix.loc[important_features, important_features]  # Subset untuk variabel penting
sns.heatmap(subset_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap Korelasi Variabel Penting')
plt.savefig('heatmap.png')
plt.show()