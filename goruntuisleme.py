import cv2
import numpy as np
import time
import math
from ultralytics import YOLOv10

# Modeli yükle
model = YOLOv10('best.pt')

# Kamera başlat
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Kamera bağlantısı kurulamadı!")
    exit()

frame_h, frame_w = frame.shape[:2]

# Grid ayarları
GRID_COLS, GRID_ROWS = 3, 3
COL_W, ROW_H = frame_w // GRID_COLS, frame_h // GRID_ROWS

# Takip değişkenleri
hedef_bulundu = False
hedef_kayip = False
kayip_zaman = 0
kayip_pozisyon = None
izleme_suresi = 0
kilitli = False
kilit_zaman = 0
vuruldu_anim = False
vuruldu_baslangic = 0
hedef_tahmini = None
hedef_id = None
pozisyonlar = []
hizlar = []
tarama_alan = [0, 0, COL_W, ROW_H]
tarama_indeks = 0

def renkler():
    return {
        'KIRMIZI': (0, 0, 255),
        'YESIL': (0, 255, 0),
        'MAVI': (255, 0, 0),
        'SARI': (0, 255, 255),
        'CYAN': (255, 255, 0),
        'MAGENTA': (255, 0, 255),
        'BEYAZ': (255, 255, 255)
    }

RENK = renkler()

# Pencere ayarları
PENCERE = "🎯 İHA Takip ve Hedefleme Sistemi 🎯"
cv2.namedWindow(PENCERE, cv2.WINDOW_NORMAL)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOYUT = 0.6
FONT_KALIN = 2

baslangic_zaman = time.time()
son_tarama = time.time()
tarama_aralik = 0.5

# Hız hesaplama

def hiz_hesapla(pozisyonlar, zamanlar):
    if len(pozisyonlar) < 2 or len(zamanlar) < 2:
        return None
    x1, y1 = pozisyonlar[-2]
    x2, y2 = pozisyonlar[-1]
    t1, t2 = zamanlar[-2], zamanlar[-1]
    dt = t2 - t1
    if dt == 0:
        return None
    return ( (x2 - x1) / dt, (y2 - y1) / dt )

# Gelecek konum tahmini

def konum_tahmin(pozisyonlar, hizlar, gecen_sure):
    if not pozisyonlar or not hizlar:
        return None
    x, y = pozisyonlar[-1]
    if isinstance(hizlar[-1], tuple) and len(hizlar[-1]) == 2:
        vx, vy = hizlar[-1]
    else:
        if len(pozisyonlar) >= 2 and len(hizlar) >= 2:
            son, onceki = pozisyonlar[-1], pozisyonlar[-2]
            t_son = hizlar[-1][0] if isinstance(hizlar[-1], tuple) else time.time()
            t_onceki = hizlar[-2][0] if isinstance(hizlar[-2], tuple) else (t_son - 0.1)
            dt = t_son - t_onceki
            if dt > 0:
                vx = (son[0] - onceki[0]) / dt
                vy = (son[1] - onceki[1]) / dt
            else:
                return None
        else:
            return None
    return (int(x + vx * gecen_sure), int(y + vy * gecen_sure))

# Ok çizimi

def ok_ciz(img, bas, son, renk, kalin=2):
    cv2.arrowedLine(img, bas, son, renk, kalin)

# Hedef kutusu çizimi

def hedef_kutusu(img, kutu, izleme_suresi, kilitli=False, vuruldu_anim=False, vuruldu_baslangic=0):
    x1, y1, x2, y2 = map(int, kutu)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    w, h = x2 - x1, y2 - y1
    # Merkez çizgileri
    cv2.line(img, (cx, 0), (cx, img.shape[0]), RENK['KIRMIZI'], 1)
    cv2.line(img, (0, cy), (img.shape[1], cy), RENK['KIRMIZI'], 1)
    # Köşe çizgileri
    kose = int(min(w, h) * 0.2)
    koseler = [
        ((x1, y1), (x1 + kose, y1)), ((x1, y1), (x1, y1 + kose)),
        ((x2, y1), (x2 - kose, y1)), ((x2, y1), (x2, y1 + kose)),
        ((x1, y2), (x1 + kose, y2)), ((x1, y2), (x1, y2 - kose)),
        ((x2, y2), (x2 - kose, y2)), ((x2, y2), (x2, y2 - kose)),
    ]
    for bas, son in koseler:
        cv2.line(img, bas, son, RENK['YESIL'], 2)
    durum = f"HEDEF KİLİTLENDİ - {izleme_suresi:.1f}s" if kilitli else f"TAKİP EDİLİYOR - {izleme_suresi:.1f}s"
    cv2.putText(img, durum, (x1, y1 - 10), FONT, FONT_BOYUT, RENK['KIRMIZI'] if kilitli else RENK['YESIL'], FONT_KALIN)
    # Vuruldu animasyonu
    if vuruldu_anim:
        sure = time.time() - vuruldu_baslangic
        if sure <= 1.0:
            alpha = 1.0 - sure
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), RENK['KIRMIZI'], -1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            cv2.putText(img, "HEDEF VURULDU!", (cx - 100, cy), FONT, 1.0, RENK['KIRMIZI'], 3)

# Kaybolan hedef için ok

def kayip_hedef_goster(img, son_poz, tahmin=None):
    cx, cy = img.shape[1] // 2, img.shape[0] // 2
    x, y = son_poz
    kenar = None
    if x < 0:
        kenar = (0, y)
    elif x >= img.shape[1]:
        kenar = (img.shape[1] - 1, y)
    elif y < 0:
        kenar = (x, 0)
    elif y >= img.shape[0]:
        kenar = (x, img.shape[0] - 1)
    if kenar:
        bas = (cx, cy)
        yon = (kenar[0] - cx, kenar[1] - cy)
        uzunluk = math.sqrt(yon[0]**2 + yon[1]**2)
        if uzunluk > 0:
            norm = (yon[0] / uzunluk, yon[1] / uzunluk)
            son = (int(cx + norm[0] * 100), int(cy + norm[1] * 100))
            ok_ciz(img, bas, son, RENK['SARI'], 3)
            if tahmin:
                cv2.putText(img, "Tahmini Konum", (son[0] + 10, son[1]), FONT, FONT_BOYUT, RENK['SARI'], FONT_KALIN)

# Grid çizimi

def grid_ciz(img, alan):
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            x, y = j * COL_W, i * ROW_H
            if [x, y, COL_W, ROW_H] == alan:
                cv2.rectangle(img, (x, y), (x + COL_W, y + ROW_H), RENK['YESIL'], 2)
            else:
                cv2.rectangle(img, (x, y), (x + COL_W, y + ROW_H), RENK['MAVI'], 1)

# YOLO tespitlerini işle

def yolo_sonuclari(results, guven=0.3):
    tespitler = []
    if hasattr(results, 'boxes') and len(results.boxes) > 0:
        for box in results.boxes:
            conf = float(box.conf)
            if conf >= guven:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls)
                tespitler.append((x1, y1, x2, y2, conf, cls_id))
    return tespitler

# Ana döngü
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera görüntüsü alınamadı!")
            break
        goster = frame.copy()
        simdi = time.time()
        if not hedef_bulundu:
            if simdi - son_tarama > tarama_aralik:
                tarama_indeks = (tarama_indeks + 1) % (GRID_COLS * GRID_ROWS)
                satir = tarama_indeks // GRID_COLS
                sutun = tarama_indeks % GRID_COLS
                tarama_alan = [sutun * COL_W, satir * ROW_H, COL_W, ROW_H]
                son_tarama = simdi
            x, y, w, h = tarama_alan
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            w = max(1, min(w, frame_w - x))
            h = max(1, min(h, frame_h - y))
            bolge = frame[y:y+h, x:x+w]
            if bolge.size == 0:
                continue
            cv2.rectangle(goster, (x, y), (x+w, y+h), RENK['YESIL'], 3)
            cv2.putText(goster, "TARAMA MODU", (x+10, y+30), FONT, FONT_BOYUT, RENK['YESIL'], FONT_KALIN)
            sonuc = model(bolge)[0]
            tespitler = yolo_sonuclari(sonuc, guven=0.7)
            for i, (x1, y1, x2, y2, conf, cls_id) in enumerate(tespitler):
                global_x1 = x + x1
                global_y1 = y + y1
                global_x2 = x + x2
                global_y2 = y + y2
                hedef_bulundu = True
                hedef_kayip = False
                hedef_id = i
                merkez = (int((global_x1 + global_x2) / 2), int((global_y1 + global_y2) / 2))
                pozisyonlar.append(merkez)
                pozisyonlar = pozisyonlar[-10:]
                cv2.rectangle(goster, (int(global_x1), int(global_y1)), (int(global_x2), int(global_y2)), RENK['KIRMIZI'], 2)
                cv2.putText(goster, f"İHA: {conf:.2f}", (int(global_x1), int(global_y1) - 10), FONT, FONT_BOYUT, RENK['KIRMIZI'], FONT_KALIN)
                break
        else:
            sonuc = model(frame)[0]
            tespitler = yolo_sonuclari(sonuc, guven=0.5)
            gorunuyor = False
            for i, (x1, y1, x2, y2, conf, cls_id) in enumerate(tespitler):
                if pozisyonlar:
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    son_x, son_y = pozisyonlar[-1]
                    mesafe = math.sqrt((cx - son_x)**2 + (cy - son_y)**2)
                    if mesafe < frame_w * 0.3:
                        gorunuyor = True
                        merkez = (int(cx), int(cy))
                        pozisyonlar.append(merkez)
                        pozisyonlar = pozisyonlar[-10:]
                        if len(pozisyonlar) >= 2:
                            if len(hizlar) > 0 and isinstance(hizlar[0], tuple) and len(hizlar[0]) == 2 and not isinstance(hizlar[0][0], tuple):
                                onceki = pozisyonlar[-2]
                                dt = 0.1
                                vx = (merkez[0] - onceki[0]) / dt
                                vy = (merkez[1] - onceki[1]) / dt
                                if len(hizlar) < 10:
                                    hizlar.append((vx, vy))
                                else:
                                    hizlar = hizlar[1:] + [(vx, vy)]
                            else:
                                if len(hizlar) < 10:
                                    hizlar.append((simdi, merkez))
                                else:
                                    hizlar = hizlar[1:] + [(simdi, merkez)]
                        if not kilitli:
                            izleme_suresi = simdi - baslangic_zaman
                            if izleme_suresi >= 4.0:
                                kilitli = True
                                kilit_zaman = simdi
                        else:
                            kilit_sure = simdi - kilit_zaman
                            if kilit_sure >= 4.0 and not vuruldu_anim:
                                vuruldu_anim = True
                                vuruldu_baslangic = simdi
                            if vuruldu_anim and (simdi - vuruldu_baslangic > 1.0):
                                vuruldu_anim = False
                                kilitli = False
                                izleme_suresi = 0
                        takip_sure = izleme_suresi
                        if kilitli:
                            takip_sure = simdi - kilit_zaman
                        hedef_kutusu(goster, (int(x1), int(y1), int(x2), int(y2)), takip_sure, kilitli, vuruldu_anim, vuruldu_baslangic)
                        break
            if not gorunuyor:
                if not hedef_kayip:
                    hedef_kayip = True
                    kayip_zaman = simdi
                    if pozisyonlar:
                        kayip_pozisyon = pozisyonlar[-1]
                    if len(pozisyonlar) >= 2:
                        son = pozisyonlar[-1]
                        onceki = pozisyonlar[-2]
                        dt = 0.1
                        vx = (son[0] - onceki[0]) / dt
                        vy = (son[1] - onceki[1]) / dt
                        if len(hizlar) < 10:
                            hizlar.append((vx, vy))
                        else:
                            hizlar = hizlar[1:] + [(vx, vy)]
                if simdi - kayip_zaman > 5.0:
                    hedef_bulundu = False
                    hedef_kayip = False
                    pozisyonlar = []
                    hizlar = []
                    izleme_suresi = 0
                    kilitli = False
                    hedef_tahmini = None
                else:
                    if kayip_pozisyon:
                        if len(pozisyonlar) >= 2 and len(hizlar) >= 1:
                            gecen = simdi - kayip_zaman
                            hedef_tahmini = konum_tahmin(pozisyonlar, hizlar, gecen)
                        kayip_hedef_goster(goster, kayip_pozisyon, hedef_tahmini)
                        cv2.putText(goster, f"HEDEF KAYIP: {simdi - kayip_zaman:.1f}s", (50, 50), FONT, 1, RENK['SARI'], 2)
                        if hedef_tahmini:
                            cv2.putText(goster, "TAHMİNİ KONUM", (50, 80), FONT, 1, RENK['SARI'], 2)
        if not hedef_bulundu:
            grid_ciz(goster, tarama_alan)
        durum = "ARAMA MODU: TARAMA" if not hedef_bulundu else ("TAKİP MODU: HEDEF KİLİTLENDİ" if kilitli else "TAKİP MODU: HEDEF İZLENİYOR")
        cv2.putText(goster, durum, (10, goster.shape[0] - 20), FONT, 1, RENK['BEYAZ'], 2)
        fps = f"FPS: {1.0 / (time.time() - simdi + 0.001):.1f}"
        cv2.putText(goster, fps, (frame_w - 150, 30), FONT, 1, RENK['BEYAZ'], 2)
        cv2.imshow(PENCERE, goster)
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Esc tuşuna basıldı... Kapatılıyor...")
            break
except Exception as e:
    print(f"Hata oluştu: {e}")
    import traceback
    traceback.print_exc()
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Program başarıyla sonlandırıldı!")