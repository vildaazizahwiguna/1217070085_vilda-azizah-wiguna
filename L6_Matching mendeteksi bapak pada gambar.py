import cv2
from matplotlib import pyplot as plt 
# panggil dan konversi warna agar sesuai dengan Matplotlib 
bapak = cv2.imread('bapak.png') 
bapak = cv2.cvtColor(bapak, cv2.COLOR_BGR2RGB) # simpan dengan nama yang sama = ditumpuk
# panggil dan konversi warna agar sesuai dengan Matplotlib
solvay = cv2.imread('sidang isbat.jpeg') 
solvay = cv2.cvtColor(solvay, cv2.COLOR_BGR2RGB) 
plt.subplot(121),plt.imshow(bapak), plt.title('Bapak') 
plt.subplot(122),plt.imshow(solvay), plt.title('Sidang Isbat') 
plt.show() 