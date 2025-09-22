# Word2Vec & ALS Recommendation System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jFH9gPYsvGAriDFPmo4QG4sVDkhlBqqz?usp=sharing)

---

# Deskripsi Singkat
Proyek ini membangun **sistem rekomendasi produk** menggunakan dua pendekatan utama:

1. **Content-Based Filtering (CBF) dengan Word2Vec**  
   Merekomendasikan produk berdasarkan kemiripan deskripsi teks produk.
2. **Collaborative Filtering (CF) dengan ALS (Alternating Least Squares)**  
   Merekomendasikan produk berdasarkan pola interaksi user–produk.
3. **Evaluasi Produk**  
   Menampilkan Top-5 dan Bottom-5 produk berdasarkan rata-rata rating.

Notebook utama: `Word2Vec&ALS.ipynb`  
Dibuat menggunakan **Google Colab** dan **Python**.

---

# Alur Kerja Sistem

# 1. Persiapan & Import Library
- Instalasi library: `numpy`, `pandas`, `gensim`, `implicit`, `matplotlib`, `seaborn`, `scikit-surprise`.
- Load dataset `amazon.csv` dengan kolom utama: `user_id`, `product_name`, `about_product`, `rating`.

# 2. Preprocessing Data
- Membersihkan kolom `about_product`. Jika kosong → fallback ke `product_description` atau `product_name`.
- Membersihkan kolom `rating` → ekstraksi angka (contoh: `"4.0 out of 5"` → `4.0`).
- Drop baris dengan nilai kosong.
- Tokenisasi deskripsi produk.

# 3. Content-Based Filtering (Word2Vec)
- Latih model Word2Vec dari token deskripsi.
- Representasi produk = rata-rata embedding kata.
- Hitung **cosine similarity** antar produk.
- Fungsi: `cb_recommend(product_name, top_k)`.
- Visualisasi: bar chart rating produk hasil rekomendasi.

# 4. Collaborative Filtering (ALS)
- Encode `user_id` dan `product_name` jadi kode numerik.
- Bentuk matrix user–item (sparse).
- Latih model ALS dengan `implicit.als.AlternatingLeastSquares`.
- Fungsi: `als_recommend(user_id, top_k)`.
- Visualisasi: bar chart skor rekomendasi.

# 5. Evaluasi Produk
- Hitung Top-5 dan Bottom-5 produk berdasarkan rata-rata rating.
- Visualisasi bar chart produk top/bottom.

---

# Output yang Dihasilkan
1. **Content-Based Recommendation**
   ```
   Rekomendasi CB untuk: Produk_A
                product_name  rating
   12        Produk_B         4.5
   87        Produk_C         4.3
   ...
   ```

2. **ALS Recommendation**
   ```
   Rekomendasi ALS untuk user: user_123
   1. Produk X → 0.1452
   2. Produk Y → 0.1321
   3. Produk Z → 0.1103
   ```

3. **Top & Bottom Products**
   ```
   Top-5 Produk:
   Produk_A → 5.0
   Produk_B → 4.8
   ...

   Bottom-5 Produk:
   Produk_X → 2.0
   Produk_Y → 2.3
   ...
   ```

4. **Visualisasi**
   - Distribusi rating (countplot).
   - Boxplot rating.
   - Bar chart rekomendasi produk.
   - Bar chart Top/Bottom produk.

---

# Cara Menjalankan
1. Upload dataset `amazon.csv` ke Google Drive.
2. Buka notebook [Word2Vec&ALS.ipynb](https://colab.research.google.com/drive/1jFH9gPYsvGAriDFPmo4QG4sVDkhlBqqz?usp=sharing).
3. Pastikan path dataset sesuai:
   ```python
   DATA_PATH = "/content/drive/My Drive/Colab Notebooks/Word2Vec&ALS/Dataset/amazon.csv"
   ```
4. Jalankan cell dari atas ke bawah.

---

# Kesimpulan
- **Word2Vec (CBF)** cocok untuk rekomendasi berbasis konten, terutama jika deskripsi produk kaya teks.  
- **ALS (CF)** lebih kuat pada dataset dengan interaksi user–produk yang besar.  
- **Evaluasi rating** memberikan insight produk terbaik & terburuk.  
