import cv2
import numpy as np
import matplotlib.pyplot as plt

# Membaca gambar
img_bgr = cv2.imread("gedung.jpg") 
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
height, width, channel = img_bgr.shape

# Membuat variabel untuk menyimpan data histogram
hgr = np.zeros((256), dtype=np.int32)
hgg = np.zeros((256), dtype=np.int32)
hgb = np.zeros((256), dtype=np.int32)

# Fungsi untuk menghitung histogram
def hitung_histogram():
    for y in range(height):
        for x in range(width):
            red = img[y, x, 0]
            green = img[y, x, 1]
            blue = img[y, x, 2]
            hgr[red] += 1
            hgg[green] += 1
            hgb[blue] += 1

# Menghitung histogram gambar
hitung_histogram()

# Fungsi untuk menampilkan hasil
def plot_hasil(red, green, blue):
    bins = np.linspace(0, 256, 256)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    for ax in [ax1, ax2, ax3]:
        ax.spines["top"].set_visible(False)
        ax.grid(color='b', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.tick_params(direction='out', color='b', width=1)
    ax1.set_title('Merah')
    ax2.set_title('Hijau')
    ax3.set_title('Biru')
    ax1.hist(red, bins, color="red", alpha=1)
    ax2.hist(green, bins, color="green", alpha=1)
    ax3.hist(blue, bins, color="blue", alpha=1)
    plt.rcParams['figure.figsize'] = [20, 7]
    plt.show()

# Menampilkan histogram
plot_hasil(hgr, hgg, hgb)
