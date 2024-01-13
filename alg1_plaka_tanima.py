import os   # Operating System, dosya ve dizinlerde kolaylıkla işlemler yapmamızı sağlar.
import cv2  # OpenCV bilgisayarla görü, makine öğrenimi, görüntü işleme gibi işlerde kullanılan açık kaynak kütüphane.
import numpy as np  # Numerical Python, yüksek performanslı sayısal işlemler yapmak için kullanılan bir Python kütüphanesidir.
import matplotlib.pyplot as plt # Veri görselleştirmesinde kullandığımız temel python kütüphanesidir.


"""
#                  **RESMİ OKUMA, GRİ TONLARINA ÇEVİRME**

resim_adres = os.listdir("/Users/miracbaysal/Desktop/AracProje/Photos/")

resim_adres.sort()

img = cv2.imread("/Users/miracbaysal/Desktop/AracProje/Photos/" + resim_adres[9]) # Klasördeki 1. resmi oku.
img = cv2.resize(img, (500, 500))   # Resmi 500e 500 olarak yeniden boyutlandırdık.

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))    # Resmin renklerini BGR'dan RGB'ye convert ettik.
plt.show()  # Resmi gösterdik.

img_bgr = img
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # BGR'dan Gray'e

plt.imshow(img_gray, cmap="gray")   # cmap = color_map kısaltması
plt.show()

#                   **GÜRÜLTÜ GİDERME, KENARLIK TESPİTİ**
# Bu işlem için medyan bulanıklaştırma yapıcaz. Medyan bulanıklaştırma, verdiğimiz filtre boyutunda ortanca değeri alarak işleme devam ediyor.

ir_img = cv2.medianBlur(img_gray, 5) # ir = işlem resmi,    burada verdiğimiz 5 değeri resmi filtreleyeceğimiz değer. 5e5 boyutlandırarak işlem sürecek.
ir_img = cv2.medianBlur(ir_img, 5) # Aynı işlemi tekrar yapıyoruz. Plaka 5 piksel aralığında olduğu için bu işlemden çok etkilenmicektir ama kalan şeylerden daha çok arınmamızı sağlicak.

plt.imshow(ir_img, cmap="gray")
plt.show()

# Yoğunluk merkezi bulma, bunun için birkaç yöntem var ortalama, medyan, tepedeğer gibi. Biz medyan kullanıcaz.
medyan = np.median(ir_img)
# Bu yoğunluk merkezinin 2/3ü alt, 3/4ü üst yoğunluk merkezi
low = 0.67 * medyan 
high = 1.33 * medyan

# John F. Canny'nin bulduğu algoritma, not John F. Kennedy
kenarlik = cv2.Canny(ir_img, low, high) 
# Canny algoritmasının çıkardığı sonuç high eşik değerinin üstünde kalırsa kenarlık kabul edilir. Üstünde kalmazsa low eşik değerine bakılır.
# Low ile High arasında kalırsa etrafındaki piksellere bakılır ve high değerinin üstünde kalırsa kenarlık kabul edilir.

plt.imshow(kenarlik, cmap="gray")
plt.show()

# Kenarlığımız tek piksel üzerinde gittiği için bunu biraz genişletmemiz gerek.
# np.ones((3,3), np.uint8), iterations=1) -->  integer tipinde pozitif tam sayılar olucak ve 8 bit olucak. 3e3 kenarlıklar şeklinde filtrelicek. iterations=1 ise kaç kere genişletme yapılacağı
kenarlik = cv2.dilate(kenarlik, np.ones((3,3), np.uint8), iterations=1) 

plt.imshow(kenarlik, cmap="gray")
plt.show()


#                    **DİKDÖRTGENİ ALMA**
# Elimizde olan son fotoğrafta, hiyerarşik yapıda olan dikdörtgenleri alıp köşegen pikellerinin değerlerini buluyoruz.
cnt = cv2.findContours(kenarlik, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.RETR_TREE --> Hiyerarşik yapıda olanları bulur.
# cv2.CHAIN_APPROX_SIMPLE  --> Köşegen pixellerini verir.
cnt = cnt[0]
# Elimize geçen contourların hepsine ihtiyacımız olmadığı için eleme işlemi yapıcaz. 
# Plakamız büyüklük olarak ilk 20 büyük contour arasında olacağından dikdörtgenlerin alanına göre sıralama yapıyoruz
cnt = sorted(cnt, key=cv2.contourArea, reverse=True)

H, W = 500, 500
plaka = None

# Aşağıda yapacağımız for döngüsü ile dikdörtgenlerin hepsini alacağız. Sonrasında arasında en az 2 kat olanları seçip bunların koordinatlarını alacağız.
#
for c in cnt:
    rect = cv2.minAreaRect(c)   #Dikdörtgen yapıdakilerin hepsini aldık. (1.Aşama)
    (x, y),(w, h), r = rect
    if(w > h and w > h*2) or (h > w and h > w*2):   #Oran en az 2  (2.Aşama)
        box = cv2.boxPoints(rect)   #[[12,13], [25,13], [20,13], [13,45]] gibi bir ifade döndürecek. Sayılar sallama
        box = np.int64(box) # Değerlerin int olması için

        # Maksimum ve minimum noktalarımızı alıyoruz.
        minx = np.min(box[:,0]) # Bütün değerleri al hepsini sadece 0.indextekileri kullan yani 73.satırda 12, 25, 20, 13 gibi değerleri alacak
        miny = np.min(box[:,1])
        maxx = np.max(box[:,0])
        maxy = np.max(box[:,1])

        # Plaka olasılığı yüksek dikdörtgenleri kesip alıyoruz. 
        olasi_plaka = img_gray[miny:maxy, minx:maxx].copy()
        olasi_medyan = np.median(olasi_plaka)

        # 3 çeşit kontrol yapacağız. yapilacaklar.txt dosyası 3. ve 4. maddeler. 3 kontrol olma sebebi bazı durumlarda h ve w değerlerinin yer değiştirmesi

        kontrol1 = olasi_medyan > 84 and olasi_medyan < 200    # Yoğunluk kontrolü (3.Aşama)
        kontrol2 = h < 50 and w < 150    # Sınır kontrolü (4.Aşama)
        kontrol3 = w < 50 and h < 150    # Sınır kontrolü (4.Aşama)

        print(f"olasi_plaka medyan:{olasi_medyan} genislik: {w} yukseklik: {h}")

        plt.figure()
        kontrol = False
        if(kontrol1 and (kontrol2 or kontrol3)):
            #plakadır
            cv2.drawContours(img, [box], 0, (0,255,0), 2)
            # img değişkenimizin çizdireceğimiz contourlarını belirttik(box). Sonrasında contourların köşegenlerden çizileceğini belirtmek için 0 değerini giriyoruz.
            # sonrasında bgr renk değerlerimizi giriyoruz. çizdirdiğimiz kısmın yeşil olmasını istediğimiz için yeşil kısmı 255 olarak ayarladık. Son adımda ise pixel kalınlığı yazıyoruz.
            
            plaka = [int(i) for i in [minx, miny, w, h]]

            plt.title("Plaka tespit edildi!!!")
            kontrol = True
        else:
            #plaka değildir
            cv2.drawContours(img, [box], 0, (0,0,255), 2)
            plt.title("Plaka tespit edilemedi!!")
            

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

        if(kontrol):
            break


#Plaka bulunmuştur!
"""

def plaka_konum_don(img):

    img_bgr = img
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # BGR'dan Gray'e


    #                   **GÜRÜLTÜ GİDERME, KENARLIK TESPİTİ**
    # Bu işlem için medyan bulanıklaştırma yapıcaz. Medyan bulanıklaştırma, verdiğimiz filtre boyutunda ortanca değeri alarak işleme devam ediyor.

    ir_img = cv2.medianBlur(img_gray, 5) # ir = işlem resmi,    burada verdiğimiz 5 değeri resmi filtreleyeceğimiz değer. 5e5 boyutlandırarak işlem sürecek.
    ir_img = cv2.medianBlur(ir_img, 5) # Aynı işlemi tekrar yapıyoruz. Plaka 5 piksel aralığında olduğu için bu işlemden çok etkilenmicektir ama kalan şeylerden daha çok arınmamızı sağlicak.

    # Yoğunluk merkezi bulma, bunun için birkaç yöntem var ortalama, medyan, tepedeğer gibi. Biz medyan kullanıcaz.
    medyan = np.median(ir_img)
    # Bu yoğunluk merkezinin 2/3ü alt, 3/4ü üst yoğunluk merkezi
    low = 0.67 * medyan 
    high = 1.33 * medyan

    # John F. Canny'nin bulduğu algoritma, not John F. Kennedy
    kenarlik = cv2.Canny(ir_img, low, high) 
    # Canny algoritmasının çıkardığı sonuç high eşik değerinin üstünde kalırsa kenarlık kabul edilir. Üstünde kalmazsa low eşik değerine bakılır.
    # Low ile High arasında kalırsa etrafındaki piksellere bakılır ve high değerinin üstünde kalırsa kenarlık kabul edilir.

    # Kenarlığımız tek piksel üzerinde gittiği için bunu biraz genişletmemiz gerek.
    # np.ones((3,3), np.uint8), iterations=1) -->  integer tipinde pozitif tam sayılar olucak ve 8 bit olucak. 3e3 kenarlıklar şeklinde filtrelicek. iterations=1 ise kaç kere genişletme yapılacağı
    kenarlik = cv2.dilate(kenarlik, np.ones((3,3), np.uint8), iterations=1) 


    #                    **DİKDÖRTGENİ ALMA**
    # Elimizde olan son fotoğrafta, hiyerarşik yapıda olan dikdörtgenleri alıp köşegen pikellerinin değerlerini buluyoruz.
    cnt = cv2.findContours(kenarlik, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.RETR_TREE --> Hiyerarşik yapıda olanları bulur.
    # cv2.CHAIN_APPROX_SIMPLE  --> Köşegen pixellerini verir.
    cnt = cnt[0]
    # Elimize geçen contourların hepsine ihtiyacımız olmadığı için eleme işlemi yapıcaz. 
    # Plakamız büyüklük olarak ilk 20 büyük contour arasında olacağından dikdörtgenlerin alanına göre sıralama yapıyoruz
    cnt = sorted(cnt, key=cv2.contourArea, reverse=True)

    H, W = 500, 500
    plaka = None

    # Aşağıda yapacağımız for döngüsü ile dikdörtgenlerin hepsini alacağız. Sonrasında arasında en az 2 kat olanları seçip bunların koordinatlarını alacağız.
    #
    for c in cnt:
        rect = cv2.minAreaRect(c)   #Dikdörtgen yapıdakilerin hepsini aldık. (1.Aşama)
        (x, y),(w, h), r = rect
        if(w > h and w > h*2) or (h > w and h > w*2):   #Oran en az 2  (2.Aşama)
            box = cv2.boxPoints(rect)   #[[12,13], [25,13], [20,13], [13,45]] gibi bir ifade döndürecek. Sayılar sallama
            box = np.int64(box) # Değerlerin int olması için

            # Maksimum ve minimum noktalarımızı alıyoruz.
            minx = np.min(box[:,0]) # Bütün değerleri al hepsini sadece 0.indextekileri kullan yani 73.satırda 12, 25, 20, 13 gibi değerleri alacak
            miny = np.min(box[:,1])
            maxx = np.max(box[:,0])
            maxy = np.max(box[:,1])

            # Plaka olasılığı yüksek dikdörtgenleri kesip alıyoruz. 
            olasi_plaka = img_gray[miny:maxy, minx:maxx].copy()
            olasi_medyan = np.median(olasi_plaka)

            # 3 çeşit kontrol yapacağız. yapilacaklar.txt dosyası 3. ve 4. maddeler. 3 kontrol olma sebebi bazı durumlarda h ve w değerlerinin yer değiştirmesi

            kontrol1 = olasi_medyan > 84 and olasi_medyan < 200    # Yoğunluk kontrolü (3.Aşama)
            kontrol2 = h < 50 and w < 150    # Sınır kontrolü (4.Aşama)
            kontrol3 = w < 50 and h < 150    # Sınır kontrolü (4.Aşama)

            print(f"olasi_plaka medyan:{olasi_medyan} genislik: {w} yukseklik: {h}")

            kontrol = False
            if(kontrol1 and (kontrol2 or kontrol3)):
                #plakadır
                cv2.drawContours(img, [box], 0, (0,255,0), 2)
                # img değişkenimizin çizdireceğimiz contourlarını belirttik(box). Sonrasında contourların köşegenlerden çizileceğini belirtmek için 0 değerini giriyoruz.
                # sonrasında bgr renk değerlerimizi giriyoruz. çizdirdiğimiz kısmın yeşil olmasını istediğimiz için yeşil kısmı 255 olarak ayarladık. Son adımda ise pixel kalınlığı yazıyoruz.
                
                plaka = [int(i) for i in [minx, miny, w, h]]    #x, y, w, h

                kontrol = True
            else:
                #plaka değildir
                # cv2.drawContours(img, [box], 0, (0,0,255), 0)
                continue

            if(kontrol):
                return plaka
    return []
