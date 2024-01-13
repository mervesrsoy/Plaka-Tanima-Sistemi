import os
import matplotlib.pyplot as plt
import cv2
from alg1_plaka_tanima import plaka_konum_don   #Kendi algoritmamızdan fonksiyonumuzu import ettik.
"""
# 1.Algoritma veri inceleme
#--------------------------

resim_adres = os.listdir("Photos")

for image_url in resim_Adres:
    img = cv2.imread("Photos/" + image_url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (500, 500))
    plt.imshow(img)
    plt.show()
"""

# 2.Algoritma veri inceleme
#--------------------------
    
resim_adres = os.listdir("Photos")

for image_url in resim_adres:
    img = cv2.imread("Photos/" + image_url)
    
    img = cv2.resize(img, (500, 500))
    plaka = plaka_konum_don(img)    #x,y,w,h değerleri geliyor.
    x, y, w, h = plaka
    if(w > h):  #Bazı durumlarda w>h bazı durumlarda tam tersi olduğu için if else'ye sokuyoruz.
        plaka_bgr = img[y:y+h, x:x+w].copy()
    else:
        plaka_bgr = img[y:y+w, x:x+h].copy()
        

    img = cv2.cvtColor(plaka_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()