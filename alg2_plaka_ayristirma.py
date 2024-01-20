import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from alg1_plaka_tanima import plaka_konum_don

veriler = os.listdir("Photos")

isim = veriler[1]

img = cv2.imread("Photos/" + isim)
img = cv2.resize(img, (500, 500))

plaka = plaka_konum_don(img)
x, y, w, h = plaka

if (w > h):
    plaka_bgr = img[y:y+h, x:x+w].copy()
else:
    plaka_bgr = img[y:y+w, x:x+h].copy()

plt.imshow(plaka_bgr)
plt.show()


# Görüntüyü aldığımızda pixellikler var bunlarda bizi hataya zorlayabilir.
# Pixellerden kurtulmak için değerlerimizi 2 kat arttıracağız. Netlikte bozulma olabilir fakat pixeller kaybolacak ve hata yapma olasılığımız azalacak.
H, W = plaka_bgr.shape[:2]
print("Orjinal boyut:", W,H)

H, W = H*2, W*2
print("Orjinal boyut:", W,H)
plaka_bgr = cv2.resize(plaka_bgr, (W,H))

plt.imshow(plaka_bgr)
plt.show()

# plaka_resim: işlem resmimiz
plaka_resim = cv2.cvtColor(plaka_bgr,cv2.COLOR_BGR2GRAY)

plt.title("Gri Format")
plt.imshow(plaka_resim, cmap="gray")
plt.show()

#               ** 1.1 EŞİKLEME İŞLEMİ **
#adaptiveTreshold = gelişmiş eşikleme işlemi
th_img = cv2.adaptiveThreshold(plaka_resim, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#1.parametre eşiklenecek resmimiz, 2.parametre eşiğin üstünde kalan pixeller 255 olarak ayarlancak,
#3.parametre Kullanılacak eşikleme türü(bizimki ortalama), 4.parametre eşiklemenin nasıl yapılacağını gösterir.
#4. parmetrede alt tarafın mı pozitif üst tarafın mı pozitif olduğunu söylemeliyiz. Biz alt tarafın pozitif olmasını istiyoruz. Bu yüzden BINARY_INV kullanacağız. Normalin tersi.
#Sebebi ise karakterlerimiz siyah renkte olduğundan siyah renklerle uğraşacağız ve renk değeri 0 lara yakın olduğu için altta kalanları pozitif ayarlayıp ona göre işlem yapmalıyız.
#5.parametre filtremizin kaç boyutlu olacağını söyler. Bu algoritmada en iyi sonuç veren değer 11, 6.parametre komşu sayısını söyler. Kaç komşunun eşik değerinin üstüne geçtiğini söyler. Burada 2 en güzel değer.

plt.title("Eşiklenmiş")
plt.imshow(th_img, cmap="gray")
plt.show()

#Gürültü yok etme işlemi

kernel = np.ones((3,3), np.uint8)
th_img = cv2.morphologyEx(th_img, cv2.MORPH_OPEN, kernel, iterations=1)
#morfolojik işlem olan açma işlemi yaptık. kernelimizi 3e3 olarak ayarladık ve 1 iterasyon yaptık.

plt.title("Gürültü yok edildi!")
plt.imshow(th_img, cmap="gray")
plt.show() 

#Contourlarımızı buluyoruz
cnt = cv2.findContours(th_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = cnt[0]
#contourlar 2 temel değişken döndürür. cnt[0] bulunduğu konumları döndürür eğer devamına .cnt[1] yazarsak hiyerarşik yapıyı döndürür.
cnt = sorted(cnt, key=cv2.contourArea, reverse=True)[:15] # Alanı büyükten küçüğe doğru ayarladık ve ilk 15'i aldık.

#1.2 ve 1.3 işlemler
for i,c in enumerate(cnt):
    rect =cv2.minAreaRect(c)
    (x, y), (w, h), r = rect

    kontrol1 = max([w,h]) < W/4
    kontrol2 = w*h > 200

    if(kontrol1 and kontrol2):
        print("karakter -> ", x, y, w, h)

        
        box = cv2.boxPoints(rect) #Sol üst, sağ üst, sol alt, sağ alt noktalar
        box = np.int64(box) # integer yaptık.

        minx = np.min(box[:,0])
        miny = np.min(box[:,1])
        maxx = np.max(box[:,0])
        maxy = np.max(box[:,1])

        odak = 2

        minx = max(0, minx-odak)
        miny = max(0, miny-odak)
        maxx = min(W, maxx+odak)
        maxy = min(H, maxy+odak)

        kesim = plaka_bgr[miny:maxy, minx:maxx].copy()

        try:
            cv2.imwrite(f"karakterseti/{isim}_{i}.jpg", kesim)
        except:
            pass

        yaz = plaka_bgr.copy()
        cv2.drawContours(yaz, [box], 0, (0, 255, 0), 1)

        plt.imshow(yaz)
        plt.show()