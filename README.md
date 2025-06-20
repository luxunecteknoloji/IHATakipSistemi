# IHATakipSistemi
YOLOv10 nesne tanıma modeli kullanarak bir İHA'yı (veya başka bir nesneyi) gerçek zamanlı kamera görüntüsü üzerinden tespit eden, takip eden ve kilitleyen bir sistemdir. 

Gerçek zamanlı olarak:

- Kameradan görüntü alır.

- Belirli alanları tarar (grid scanning).

- YOLOv10 ile hedef tespiti yapar.

- Hedefe kilitlenir ve onu takip eder.

- Kaybolursa yönünü ve tahmini konumunu gösterir.

- Hedefe 4 saniye kilitlenince "vuruldu" animasyonu oynatır.

Kurulum ve Başlatma:
- YOLOv10 modeli best.pt dosyasından yüklenir.

- Kamera başlatılır.

- Görüntü boyutuna göre 3x3 tarama alanı (grid) oluşturulur.

Takip ve Kilitleme Değişkenleri:
- Hedefin bulunup bulunmadığı (target_found)

- Kaybedildi mi, ne zaman kaybedildi vs.

- Takip süresi ve kilitlenme durumu

- Hedef pozisyonları ve hız vektörleri tutulur

Yardımcı Fonksiyonlar:
- calculate_velocity: Hedefin hızını hesaplar.

- predict_future_position: Hedef kaybolduğunda tahmini konumu hesaplar.

- draw_targeting_box: Hedef kutusunu çizer ve vuruldu animasyonu ekler.

- draw_lost_target_indicator: Hedef kaybolursa yönünü okla gösterir.

- draw_scanning_grid: Tarama grid'ini ekrana çizer.

- process_yolo_detections: YOLO çıktısını işler.



- ESC tuşuyla çıkılır.
