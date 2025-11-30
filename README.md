# NeiPalette – Renk Uyumu ve Öneri Sistemi

Bu proje, giyim ürünlerinin görsellerinden **dominant renk paletlerini** çıkarıp,
ürünler arası **renk uyumu** tahmini ve basit bir **kombin öneri sistemi** geliştirmeyi amaçlar.

Veri seti olarak DeepFashion2 benzeri bir büyük giyim veri seti düşünülmüştür.
Renk vektörleri ile klasik benzerlik bazlı bir **baseline** ve görsel + renk bilgisini birleştiren
**derin öğrenme tabanlı hibrit bir model** karşılaştırılır.

## Genel Mimari

- **Veri & Renk Özellikleri (Nehir)**
  - Ürün görselleri ve segmentasyon maskeleri ile çalışır.
  - Her ürün için 3–5 adet dominant renk çıkarır (RGB + CIELAB).
  - Bu renkler, modelde kullanılmak üzere `colors.csv` benzeri bir dosyada saklanır.

- **Hibrit Model & Eğitim (Sertaç)**
  - Backbone: ResNet-50 (veya benzeri).
  - Renk vektörünü küçük bir MLP ile embed eder.
  - Görsel embedding + renk embedding birleştirilerek:
    - Uyum skoru (regresyon),
    - Uyumlu / uyumsuz sınıfı (binary sınıflandırma)
    üretir.

- **Baseline & Öneri Sistemi (Hamza)**
  - Renk vektörleri arasında benzerlik (cosine / öklid) ile basit bir skor üretir.
  - Eşik tabanlı bir baseline sınıflandırıcı kurar.
  - Eğitilmiş hibrit modeli kullanarak seçilen ürün için en uyumlu kombinleri önerir.

## Klasör Yapısı

```text
neipalette/
  data/                 # Büyük veri seti burada (git'e eklenmez)
  notebooks/            # Jupyter defterleri
  src/                  # Python kaynak kodları
    data/
    models/
    utils/
  results/              # Deney sonuçları ve figürler
  report/               # Proje raporu (PDF)
  requirements.txt
  README.md
  .gitignore
```

## Kurulum

```bash
git clone <bu-reponun-linki>
cd neipalette

# Sanal ortam (opsiyonel ama tavsiye edilir)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Veri Hazırlığı

1. DeepFashion2 (veya seçtiğiniz veri seti) görsellerini ve anotasyonlarını indirin.
2. Lokalde `data/deepfashion2/` altına yerleştirin.
3. `notebooks/01_data_exploration.ipynb` defterini çalıştırarak:
   - Örnek görselleri inceleyin.
   - Dominant renkleri çıkarın.
   - Sonuçları `data/processed/colors.csv` gibi bir dosyaya kaydedin.

## Notebooklar

- `01_data_exploration.ipynb` – Veri keşfi + dominant renk çıkarımı.
- `02_baseline_color_similarity.ipynb` – Renk vektörleri ile baseline benzerlik ve metrikler.
- `03_hybrid_model_training.ipynb` – Hibrit modelin eğitimi ve değerlendirilmesi.
- `04_recommendation_demo.ipynb` – Eğitilmiş modelle kombin öneri demosu.

## Çalıştırma Örneği

```bash
# 1) Veri keşfi
jupyter notebook notebooks/01_data_exploration.ipynb

# 2) Baseline deneyler
jupyter notebook notebooks/02_baseline_color_similarity.ipynb

# 3) Hibrit model eğitimi
jupyter notebook notebooks/03_hybrid_model_training.ipynb

# 4) Öneri sistemi demosu
jupyter notebook notebooks/04_recommendation_demo.ipynb
```

## Gelecek Çalışmalar (Örnek fikirler)

- Renk dışında desen, kumaş, stil gibi özellikleri de modele dahil etmek.
- Kullanıcı tercihleri ve geçmiş satın alma davranışlarını kullanarak kişiselleştirme.
- Web arayüzü ile gerçek zamanlı kombin öneri prototipi geliştirmek.
- E-ticaret ortamında A/B testleri ile öneri sisteminin iş etkisini ölçmek.
