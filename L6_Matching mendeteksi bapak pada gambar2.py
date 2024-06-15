import cv2 
import numpy as np 
from matplotlib import pyplot as plt 

# Membaca gambar
img = cv2.imread('sidang isbat.jpeg', 0) 
img2 = img.copy() 
template = cv2.imread('bapak.png', 0) 
w, h = template.shape[::-1] 

# Semua metode untuk perbandingan dalam sebuah daftar
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'] 

# Memperbesar ukuran hasil plotting
plt.rcParams["figure.figsize"] = (20, 20) 

# Menyiapkan figure dengan subplots
fig, axs = plt.subplots(len(methods), 2, figsize=(20, 15))

for i, met in enumerate(methods): 
    img = img2.copy() 
    method = eval(met) 

    # Menggunakan template matching
    res = cv2.matchTemplate(img, template, method) 

    # Mencari ukuran citra template untuk menggambar kotak
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) 

    # Metode TM_SQDIFF dan TM_SQDIFF_NORMED menggunakan persamaan yang sedikit berbeda
    # sehingga dibuatkan fungsi khusus untuk mengambil nilai minimum 
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc 
    else:
        top_left = max_loc 

    bottom_right = (top_left[0] + w, top_left[1] + h) 

    # Membuat persegi pada lokasi yang ditemukan 
    cv2.rectangle(img, top_left, bottom_right, 255, 2) # 2 adalah ketebalan garis kotak 

    # Menampilkan hasil
    axs[i, 0].imshow(res, cmap='gray') 
    axs[i, 0].set_title(f'Hasil matching {met}') 
    axs[i, 0].set_xticks([])
    axs[i, 0].set_yticks([]) 

    axs[i, 1].imshow(img, cmap='gray') 
    axs[i, 1].set_title(f'Detected Point {met}') 
    axs[i, 1].set_xticks([])
    axs[i, 1].set_yticks([]) 

plt.tight_layout()
plt.show()
