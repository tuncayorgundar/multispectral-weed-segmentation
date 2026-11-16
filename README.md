# Multispektral İHA görüntülerinde mısır ve yabancı ot bölütlemesi

WeedsGalore veri setinde her pikseli arka plan, mısır veya yabancı ot olarak sınıflandıran bir DeepLabV3+ modeli. Üç soruya bakıldı:

1. RGB kamera yeterli mi, multispektral (MSI) sensör ne kazandırıyor?
2. Az görülen ot türleri kayıp fonksiyonu seçimiyle iyileştirilebiliyor mu?
3. Uçuş yüksekliği (GSD) arttıkça başarım nasıl değişiyor?

## Sonuçlar (test seti)

| Model | mIoU (%) | Not |
| --- | --- | --- |
| RGB, 3 sınıf | 81,2 | makale: 79,33 |
| MSI, 3 sınıf | **83,4** | makale: 82,90 |
| MSI, 6 sınıf, focal + Dice | **57,7** | makale: 55,52 |

- MSI, RGB'ye göre +2,2 mIoU sağlıyor. Fark en çok sezon başında (25 Mayıs, +10) ve mısırın ot sanılma oranında (%6,2 -> %3,0) görülüyor.
- Focal + Dice kaybı küçük nesneli "diğer ot" sınıfını belirgin şekilde iyileştiriyor. Quickweed ise amaranth ile spektral olarak çok benzer olduğu için %21 civarında kalıyor.
- GSD 10 mm'ye (~20 m uçuş) kadar başarım korunuyor, 20 mm'de belirgin şekilde düşüyor.

Ayrıntılı tablolar ve görseller `results/` klasöründe.

## Klasör yapısı

```
common.py                  veri hazırlama, Dataset, model, kayıp, metrik, eğitim döngüsü
notebooks/
  01_kesif.ipynb           veri keşfi (EDA)
  02_baseline.ipynb        Dataset, model ve E0 temel modeli
  03_deneyler.ipynb        E1 girdi türü, E2 kayıp fonksiyonu, E3 GSD
  04_degerlendirme.ipynb   test değerlendirmesi, karmaşıklık matrisleri, hata haritaları
results/                   grafikler, metrik tabloları, eğitim geçmişleri
```

## Çalıştırma

Notebook'lar Google Colab'da (T4 GPU) çalıştırıldı.

1. Bu klasörü Google Drive'da `MyDrive/multispectral-weed-segmentation` olarak oluşturup `common.py` dosyasını içine koyun.
2. Notebook'ları sırayla açın. Veri ilk çalıştırmada otomatik indirilir (~340 MB).
3. Bir eğitim T4'te yaklaşık 13-14 dakika sürüyor. 03'teki deneylerin tamamı birkaç saat alıyor; tamamlanan koşular tekrar eğitilmiyor.

Eğitilmiş model ağırlıkları boyut nedeniyle depoya eklenmedi.

| Notebook | Colab |
| --- | --- |
| 01_kesif | [Aç](https://colab.research.google.com/github/KULLANICI/multispectral-weed-segmentation/blob/main/notebooks/01_kesif.ipynb) |
| 02_baseline | [Aç](https://colab.research.google.com/github/KULLANICI/multispectral-weed-segmentation/blob/main/notebooks/02_baseline.ipynb) |
| 03_deneyler | [Aç](https://colab.research.google.com/github/KULLANICI/multispectral-weed-segmentation/blob/main/notebooks/03_deneyler.ipynb) |
| 04_degerlendirme | [Aç](https://colab.research.google.com/github/KULLANICI/multispectral-weed-segmentation/blob/main/notebooks/04_degerlendirme.ipynb) |

## Veri seti

WeedsGalore (GFZ Potsdam), CC BY 4.0 lisanslı: [veri](https://doidata.gfz.de/weedsgalore_e_celikkan_2024/) · [kod](https://github.com/GFZ/weedsgalore)

> Celikkan, E. vd. (2025). WeedsGalore: A multispectral and multitemporal UAV-based dataset for crop and weed segmentation in agricultural maize fields. *WACV 2025.*

## Lisans

Kod MIT lisansıyla paylaşılmıştır (`LICENSE`). Veri seti ve veriden üretilen görseller CC BY 4.0 kapsamındadır; kullanırken yukarıdaki kaynağa atıf yapılmalıdır.
