# Telco Customer Churn Tahmini

Türkiye Yapay Zeka Akademisi Makine Öğrenmesi Final Ödevi için hazırlanmış
konsol tabanlı uçtan uca makine öğrenmesi projesidir.

## Projenin amacı

Bir telekom şirketindeki müşterilerin hizmetten ayrılıp ayrılmayacağını
(`Churn`) tahmin etmektir. Problem bir **ikili sınıflandırma** problemidir:

- `0`: Müşteri ayrılmadı
- `1`: Müşteri ayrıldı

Model karşılaştırmasında churn sınıfını kaçırmamak ve yanlış alarm üretmemek
arasındaki dengeyi gösteren F1 skoru ana metrik olarak kullanılmıştır.

## Veri seti

Telco Customer Churn veri setinde müşterilerin demografik bilgileri, kullandığı
hizmetler, sözleşme tipi, aylık ücretleri, toplam ücretleri ve churn durumu
bulunur. Veri seti IBM'in açık kaynak deposundan alınmaktadır:

<https://github.com/IBM/telco-customer-churn-on-icp4d>

CSV dosyası `data/Telco-Customer-Churn.csv` altında yoksa program ilk
çalıştırmada otomatik olarak indirilir. Veri seti 7.000'den fazla müşteri
kaydına sahiptir.

## Projede uygulanan adımlar

1. Projenin amaç, kütüphane ve çalıştırma bilgilerini içeren Python docstring'i
2. Pandas ile veri setini okuma
3. Hedef değişken ve problem türü tanımı
4. İlk satırlar, boyut, veri tipleri ve temel istatistiklerin incelenmesi
5. Eksik değer kontrolü ve pipeline içinde median/most-frequent imputasyon
6. Kategorik değişkenlerin `OneHotEncoder` ile dönüştürülmesi
7. IQR yöntemiyle aykırı değerlerin incelenmesi ve eğitim kümesinden öğrenilen
   sınırlarla sınırlandırılması
8. Sayısal değişkenlerin `StandardScaler` ile ölçeklenmesi
9. `AverageMonthlySpend` ve `ServiceCount` özniteliklerinin üretilmesi
10. `SelectPercentile` ve ANOVA F-skoru ile öznitelik seçimi
11. Stratify kullanılarak train, validation ve test kümelerinin oluşturulması
12. Logistic Regression, KNN, Decision Tree ve Random Forest modellerinin
   eğitilmesi
13. Validation metrikleriyle model karşılaştırması
14. En iyi validation modeline `GridSearchCV` ile hiperparametre ayarı
15. Test kümesinde accuracy, precision, recall, F1 ve confusion matrix
   raporlaması
16. Model, önemli değişkenler ve sınırlılıkların yorumlanması
17. En iyi model için feature importance/katsayı/permutation importance
   açıklaması

## Çalıştırma

Python 3.10 veya daha yeni bir sürüm gerekir.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Bağımlılıkları kurun:

```bash
pip install -r requirements.txt
```

Programı çalıştırın:

```bash
python telco_churn_analysis.py
```

Program tüm veri inceleme ve model sonuçlarını konsola yazdırır. İnternet
bağlantısı yoksa CSV dosyası daha önce `data/` klasörüne indirilmiş olmalıdır.

## Sonuç yorumu

Çalıştırma sonunda validation F1 skoru en yüksek olan model seçilir. Bu model
Grid Search ile tekrar ayarlanır ve daha önce kullanılmayan test kümesinde
değerlendirilir. Test çıktısındaki confusion matrix, ayrılan ve ayrılmayan
müşterilerin ne kadar doğru sınıflandırıldığını gösterir.

Model başarısı, geçmiş davranış kayıtlarındaki örüntülere bağlıdır. Veri tek bir
telekom sağlayıcısından geldiği için başka şirketlere doğrudan genellenemez.
Ayrıca feature importance değerleri ilişkinin gücünü gösterebilir; tek başına
neden-sonuç ilişkisi kanıtlamaz.