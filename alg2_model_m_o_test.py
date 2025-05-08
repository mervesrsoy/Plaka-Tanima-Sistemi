import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Resimlerin özniteliklerini çıkartıyor.
def islem(img):
    
    yeni_boy = img.reshape((1600,5,5))
    orts = []
    for parca in yeni_boy:
        ort = np.mean(parca)
        orts.append(ort)
    orts = np.array(orts)
    orts = orts.reshape(1600,)
    return orts


# Verileri okuduğumuz kod bloğu
path = "karakterseti/"
siniflar = os.listdir(path)
tek_batch = 0

urls = []
sinifs = []
for sinif in siniflar:
    resimler = os.listdir(path+sinif)  
    for resim in resimler:
        urls.append(path+sinif+"/"+resim)
        sinifs.append(sinif)
        tek_batch+=1

df = pd.DataFrame({"adres":urls,"sinif":sinifs})

sinifs = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "A": 10,
    "B": 11, "C": 12, "D": 13, "E": 14, "F": 15, "G": 16, "H": 17, "I": 18, "J": 19, "K": 20,
    "L": 21, "M": 22, "N": 23, "O": 24, "P": 25, "Q": 26, "R": 27, "S": 28, "T": 29, "U": 30,
    "V": 31, "W": 32, "X": 33, "Y": 34, "Z": 35, "arkaplan": 36
}

# Kaydettiğimiz rfc dosyamızı load fonksiyonuyla okuyacağız.
dosya = "rfc_model.rfc"
rfc = pickle.load(open(dosya,("rb"))) #Read Byte


index = list(sinifs.values())   # 0 ile 36 arası indexlerimiz
siniflar = list(sinifs.keys())

df = df.sample(frac=1)  #dataframemimizi karıştırıyoruz.

for adres, sinif in df.values:
    image = cv2.imread(adres, 0)
    resim = cv2. resize(image, (200,200))
    resim = resim/255
    oznitelikler = islem(resim)

    sonuc = rfc.predict([oznitelikler])[0]
    print("sonuc: ",sonuc)

    ind = index.index(sonuc)    #random forest tarafından dönen sonucun index numarasını aldık.
    sinif = siniflar[ind]   #siniflar listesinden gerekli olan veriyi aldık.
    plt.imshow(resim,cmap="gray")   #Alınan veri gri formata dönüştür
    plt.title(f"fotoğraftaki karakter: {sinif}")    #Fotoğraftaki karakterin bulunduğu sınıf karakterin adını başlık olarak yaz.
    plt.show()