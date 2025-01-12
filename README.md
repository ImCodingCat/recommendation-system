# Laporan Proyek Machine Learning - Muhammad Dava Pasha (mdavap) @ Dicoding

## Project Overview

Di era digital yang semakin berkembang, industri restoran menghadapi tantangan yang semakin kompleks. Persaingan yang ketat antar restoran, perubahan perilaku konsumen, 

Untuk mengatasi situasi ini, restoran perlu mengadopsi strategi inovatif dalam menarik dan mempertahankan pelanggan. Salah satu solusi yang menjanjikan adalah implementasi sistem rekomendasi berbasis teknologi. Sistem ini memanfaatkan data preferensi pelanggan, riwayat pemesanan, dan pola konsumsi untuk memberikan rekomendasi menu yang personal dan relevan kepada setiap pelanggan.

Penggunaan sistem rekomendasi tidak hanya membantu meningkatkan pengalaman pelanggan, tetapi juga dapat mendorong pembelian berulang dan meningkatkan loyalitas pelanggan. Dengan memahami preferensi pelanggan secara lebih mendalam, restoran dapat menyajikan rekomendasi yang tepat sasaran, yang pada akhirnya dapat meningkatkan konversi penjualan dan pendapatan restoran.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Project ini penting diselesaikan karena dengan ada nya rekomendasi sistem restoran restoran akan mendapatkan banyak keuntungan karena pelanggan dengan sendirinya datang.
- [H. Ko, S. Lee, Y. Park, and A. Choi, "A Survey of Recommendation Systems: Recommendation Models, Techniques, and Application Fields," Electronics, vol. 11, no. 1, p. 141, Jan. 2022, doi: 10.3390/electronics11010141.](https://www.mdpi.com/2079-9292/11/1/141)

## Business Understanding

Sulitnya pelanggan datang kepada restoran dikarenakan tersebut mungkin kurang terkenal oleh sebab itu dengan rekomendasi sistem restoran-restoran tersebut akan menerima pelanggan sesuai kebutuhan pelanggan.

### Problem Statements

Menjelaskan pernyataan masalah:
- Bagaimana meningkatkan keuntungan restoran?
- Bagaimana membuat restoran lebih terkenal?

### Goals

Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
- Cara meningkatkan keuntungan restoran bisa menggunakan rekomendasi sistem dimana pelanggan akan datang dengan sendirinya.
- Dengan menggunakan rekomendasi sistemlah restoran akan lebih dikenal.

**Solution Statements**:
1. Content based filtering
    - Solusi yang pertama ialah menggunakan metode Content based filtering dimana dengan membuat sebuah vektor yang berisi kesamaan-kesamaan jenis makanan pada restoran-restoran.
    - Dengan vektor tersebut metode Content based filtering dapat dengan mencari kesamaan restoran dalam vektor tersebut.
2. Collaborative Filtering
    - Solusi yang kedua ialah Collaborative Filtering sebuah metode untuk memberikan rekomendasi kepada pengguna berdasarkan bagaimana pengguna lain yang memiliki preferensi dan perilaku serupa telah berinteraksi dengan suatu item.

## Data Understanding
Data yang akan digunakan akan di ambil dari kaggle yaitu [NYC Restaurants Data - Food Ordering and Delivery](https://www.kaggle.com/datasets/ahsan81/food-ordering-and-delivery-app-dataset). Dataset tersebut memiliki 9 kolom, jumlah data yaitu 1898 dan dataset ini tidak memiliki missing value, duplicate value ataupun outlier.

Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- order_id: Order id yang unik
- customer_id: Id dari pelanggan
- restaurant_name: Nama dari restoran
- cuisine_type: Jenis masakan
- cost: Harga dari suatu transaksi
- day_of_the_week: Berisi tentang transaksi dimana di pesan di weekday atau weekend
- rating: Rating yang diberikan oleh pelanggan
- food_preparation_time: Waktu yang dibutuhkan sebuah restoran untuk menyiapkan makanan, dihitung dalam satuan menit.
- delivery_time: Waktu yang dibutuhkan pengiriman untuk sampai kepada pelanggan, dihitung dalam satuan menit.



**Rubrik/Kriteria Tambahan (Opsional)**:
- Grafik dibawah menunjukan bahwa kebanyakan jenis makanan yang sering orang-orang pesan adalah American, Japanese dan Italian.
![](https://i.imgur.com/rpVIR9b.png)
- Grafik dibawah menunjukan jumlah pendapatan yang didapat dengan berbagai jenis makanan yang mana American yaitu 9530, Japanese yaitu 7663 dan Italian yaitu 4892.
![](https://i.imgur.com/5xQp7Q1.png)


## Data Preparation
Data preparation dilakukan adalah mencari jumlah duplikasi data, data kosong dan data yang kurang relavan. Pada kolom rating menemukan bahwa tidak setiap pelanggan memberikan sebuah rating oleh karena itu mengubah `Not given` menjadi rating 0 supaya bisa lebih relavan serta mengubah tipe data `rating` menjadi float.

Data preparation untuk Content Based Filtering
- Melakukan vectoring menggunakan `TfidfVectorizer` pada `cuisine_type` atau jenis makanan.
- Melakukan cosine_similarity dari hasil transform `TfidfVectorizer`.
- Selanjutnya, kita memetakan vector dari `cosine_similarity` di DataFrame baru dengan index dan columns menggunakan `restaurant_name` atau nama-nama restoran.

Data preparation untuk Collaborative Filtering
- Melakukan normalisasi atau mengubah angka rating menjadi kisaran 0 sampai 1 dengan cara membagi rating dengan value rating yang maksimal semisal rating 1 dibagi 5 menjadi 0.2.
- Melakukan encoding untuk customer_id dengan menginterasi `list_customerid` dan didapatkan dengan mencari customer id yang unik menjadikannya sebuah list setelah itu iterasi lagi untuk membuat dictonary dengan index yaitu menggunakan customer_id sebenarnya dan untuk value akan menggunakan index dari iterasi `list_customerid`.
- Sama halnya untuk encoding `restaurant_name` namun dengan tambahan membuat list baru dengan membalikan index menjadi index dari hasil iterasi dan value dari hasil iterasi dengan kata lain index dan value adalah kebalikan dari hasil encoding `resto_to_id_encoded`.
- Setelah melakukan normalisasi dan encoding, selanjutnya membagi dataset dengan perbandingan 9:1 atau 90% data latih dan 10% data uji, dikarenakan data yang digunakan tergolong sedikit selanjutnya membaginya dengan perbandingan tersebut.


**Rubrik/Kriteria Tambahan (Opsional)**: 
- Proses data preparation yang dilakukan yaitu mengubah sebuah rating yang kosong menjadi `Not given` dan mengubah data tipe rating menjadi float.
- Selanjutnya untuk Content Based Filtering, disini menerapkan teknik vektorisasi menggunakan `TfidfVectorizer` pada `cuisine_type` dan `cosine_similarity` dari hasil tersebut.
- Selanjutnya untuk Collaborative Filtering, menormalisasi untuk rating, encoding untuk `customer_id` dan `restaurant_name`, dan yang terakhir melakukan pembagian data latih dan data uji.
- Tahapan-tahapan tersebut sangatlah penting karena dengan memperiksa data duplikasi, data kosong dan data yang relavan akan membuat proses dan proyek akan memperoleh hasil yang akurat dan aktual.
- Tahapan-tahapan tersebut sangatlah penting karena dengan mengubah data kosong pada Rating akan mendapatkan semua data yang relavan serta melakukan teknik vektorisasi juga penting untuk bisa dapat melakukan Content Based Filtering dan samahalnya dengan normalisasi dan encoding itu dibutuhkan dikarenakan model keras hanya bisa membaca data-data yang berupa angka.

## Modeling

- Content Based Filtering
    - Content Based Filtering adalah sistem rekomendasi berbasis konten dikasus ini merekomendasikan restoran yang mirip dengan restoran lainnya.
    - Modeling:
        - Menggunakan `TfidfVectorizer` untuk mengvektorisasi `cuisine_type` atau jenis-jenis makanan
        - Selanjutnya menggunakan fungsi `cosine_similarity` dari library scikit-learn untuk menghasilkan restoran-restoran yang memiliki jenis makanan yang sama.
        - Setelah mendapatkan hasil dari `cosine_similarity` selanjutnya membuat DataFrame yang mana index dan columns menggunakan `restaurant_name` atau nama restoran.
        - Dan terakhir adalah membuat fungsi `get_resto_recommendation` dengan parameter `name` sebagai nama restoran dan `n` untuk top-N atau hasil banyak rekomendasi.
        - Fungsi ini mencari restoran-restoran yang memiliki tipe masakan (cuisine_type) yang paling mirip dengan restoran acuan menggunakan metrik cosine similarity.

- Collaborative Filtering
    - Collaborative Filtering adalah teknik rekomendasi yang merekomendasikan item berdasarkan preferensi atau perilaku pengguna lain yang memiliki kemiripan. contohnya adalah "jika pengguna A dan B memiliki preferensi yang mirip di masa lalu, maka mereka kemungkinan akan memiliki preferensi yang mirip juga di masa depan."
    - Collaborative Filtering memiliki 2 pendekatan yaitu User Based, merekomendasikan item berdasarkan pengguna yang mirip dan Item-based merekomendasikan item berdasarkan kemiripan dengan item yang disukai pengguna.
    - Pada kasus ini Collaborative Filtering menggunakan pendekatan User Based.
    - Modeling:
        - Pada tahapan modeling pada Collaborative Filtering disini menggunakan keras dari referensi [Collaborative Filtering for Movie Recommendations](https://keras.io/examples/structured_data/collaborative_filtering_movielens/).
        - Model ini menggunakan teknik embedding untuk merepresentasikan user dan restaurant dalam ruang vektor berdimensi 50 (embedding_size=50)
        - Ada 4 komponen utama dalam model:
          - user_embedding: Layer yang mengubah ID user menjadi vektor dense berdimensi 50.
          - movie_embedding: Layer yang mengubah ID restaurant menjadi vektor dense berdimensi 50.
          - user_bias: Bias term untuk setiap user.
          - movie_bias: Bias term untuk setiap restaurant.
        - Loss Function:
            - Menggunakan Binary Cross Entropy (BinaryCrossentropy).
            - Cocok untuk kasus binary classification atau prediksi rating yang dinormalisasi ke rentang 0-1.
            - Mengukur seberapa jauh prediksi model dari nilai rating sebenarnya.
        - Optimizer:
            - Menggunakan Adam optimizer dengan learning rate 0.001.
            - Adam menggabungkan keuntungan dari:
                - Momentum: membantu mengatasi local minima.
                - RMSprop: adaptif learning rate untuk setiap parameter.
            - Learning rate 0.001 adalah nilai yang cukup standar untuk memberikan convergence yang stabil.
        - Regularisasi:
            - Menggunakan L2 regularization untuk mencegah overfitting.
            - Nilai 1e-6 adalah penalty weight yang kecil untuk regularisasi.
        - Metrics:
            - Menggunakan Root Mean Squared Error (RMSE) untuk evaluasi.
            - RMSE menunjukkan rata-rata deviasi prediksi dari nilai sebenarnya dalam skala yang sama dengan data asli.
        - Dengan parameter `num_customers` sebagai jumlah dari customer, `num_restorant` sebagai jumlah restoran, dan yang terakhir `embedding_size` yang berarti ukuran dari embedding disini akan menggunakan `50`.
        - Model ini mengimplementasikan matrix factorization untuk collaborative filtering, di mana:
            - User dan restaurant direpresentasikan sebagai vektor dalam ruang latent.
            - Dot product dari vektor-vektor ini ditambah bias menghasilkan prediksi rating.
            - Model belajar representasi yang optimal melalui proses training dengan meminimalkan binary cross entropy loss.

Result Content Based Filtering with 5 Random restaurant and top-3
![](https://i.imgur.com/ky2Awjv.png)

Result Collaborative Filtering with 5 Random customer and top-3
![](https://i.imgur.com/dsucrXR.png)

## Evaluation
Content Based Filtering:
- Evaluasi untuk Content Based Filtering menggunakan Precision@k dan Recall@k
- Secara keseluruhan, sistem tampaknya bekerja secara konsisten dengan sebagian besar kasus mencapai `0.67` untuk kedua metrik, kecuali untuk kasus pertama yang memiliki presisi sempurna. Rekomendasi tersebut tampaknya mempertimbangkan jenis masakan dan gaya restoran dalam memberikan saran.

Collaborative Filtering: 
- Evaluasi model untuk Collaborative Filtering menggunakan Root Mean Squared Error dengan hasil akhir yaitu `0.3380` dan pada kasus ini menggunakan data yang sedikit oleh karena itu model mengalami overfit dikarenakan rata-rata pelanggan hanya membeli satu kali.

Business Understanding and Goal:
- Bagaimana meningkatkan keuntungan restoran?
    - Dengan menggunakan dua pendekatan diatas yaitu Content Based Filtering dan Collaborative Filtering, restoran yang memiliki jenis makanan yang sama maka pelanggan akan datang dengan sendirinya dan akan meningkatkan keuntungan restoran.
- Bagaimana membuat restoran lebih terkenal?
    - Setelah membuat dua pendekatan diatas, cara restoran lebih terkenal adalah membuat jenis makanan yang populer seperti jenis makanan Chinese, American dan seterusnya dengan begitu restoran tersebut akan muncul dalam sistem rekomendasi dan pelanggan akan datang dengan sendirinya.
- Apakah pendekatan ini berhasil mencapai goals?
    - Ya, dua pendekatan sistem rekomendasi seperti Content Based Filtering dan Collaborative Filtering dapat meningkatkan keuntungan suatu restoran karena pelanggan akan datang dengan sendirinya sesuai preferensi dan dorongan dari sistem rekomendasi tersebut dan alangkah baiknya restoran tersebut membuat jenis makanan yang sering pelanggan gemari.

Evaluasi menunjukkan bahwa sistem rekomendasi dapat membantu restoran yang kurang terkenal menarik pelanggan dengan memberikan saran yang relevan berdasarkan kebutuhan pelanggan. Dengan Content-Based Filtering, pelanggan dapat menemukan restoran yang sesuai dengan preferensi mereka berdasarkan jenis masakan dan gaya restoran, sementara Collaborative Filtering dapat memanfaatkan data pembelian pelanggan sebelumnya untuk memberikan rekomendasi yang lebih personal.

Namun, hasil evaluasi menunjukkan bahwa sistem perlu memperbaiki overfitting pada Collaborative Filtering, misalnya dengan menggunakan lebih banyak data transaksi agar rekomendasi lebih general dan akurat. Secara keseluruhan, sistem dapat meningkatkan visibilitas restoran yang kurang dikenal dan mendukung peningkatan kunjungan pelanggan.


**Rubrik/Kriteria Tambahan (Opsional)**:

Content Based Filtering:
- Precision@k adalah Proporsi item relevan di antara k item teratas yang direkomendasikan.

![](https://i.imgur.com/eX8tCh7.png)

- Recall@k adalah Proporsi item relevan yang berhasil ditemukan dari semua item relevan yang tersedia, dalam k item teratas.

![](https://i.imgur.com/k8BjoF3.png)


Collaborative Filtering: 
- Mean Absolute Error (MAE), MAE adalah nilai rata-rata dari kesalahan dengan nilai absolut antara nilai sebenarnya dan nilai prediksi.
- Root Mean Squared Error (RMSE), RMSE adalah akar kuadrat dari MSE.
- RMSE adalah akar kuadrat dari MSE sehingga lebih mudah untuk diinterpretasikan.