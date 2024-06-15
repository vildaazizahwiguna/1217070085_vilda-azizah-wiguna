import matplotlib.pyplot as plt 
from skimage.feature import graycomatrix, graycoprops 
from skimage import data 

PATCH_SIZE = 21

# Memuat gambar
image = data.camera() 

# Lokasi patch untuk rumput
grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)] 
grass_patches = [] 
for loc in grass_locations:
    grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE]) 

# Lokasi patch untuk langit
sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)] 
sky_patches = [] 
for loc in sky_locations:
    sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE]) 

# Menghitung GLCM
xs = [] 
ys = [] 
for patch in (grass_patches + sky_patches):
    glcm = graycomatrix(patch, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    xs.append(graycoprops(glcm, 'dissimilarity')[0, 0]) 
    ys.append(graycoprops(glcm, 'correlation')[0, 0]) 

# Menampilkan gambar asli dengan lokasi patch
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(3, 2, 1) 
ax.imshow(image, cmap=plt.cm.gray, vmin=0, vmax=255) 
for (y, x) in grass_locations: 
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs') 
for (y, x) in sky_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs') 
ax.set_xlabel('Gambar Asli') 
ax.set_xticks([]) 
ax.set_yticks([]) 
ax.axis('image') 

# Plot (dissimilarity, correlation) 
ax = fig.add_subplot(3, 2, 2) 
ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go', label='Rumput') 
ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo', label='Langit') 
ax.set_xlabel('GLCM Dissimilarity') 
ax.set_ylabel('GLCM Correlation') 
ax.legend() 

# Menampilkan patch rumput
for i, patch in enumerate(grass_patches): 
    ax = fig.add_subplot(3, len(grass_patches), len(grass_patches)*1 + i + 1) 
    ax.imshow(patch, cmap=plt.cm.gray, vmin=0, vmax=255) 
    ax.set_xlabel('Rumput %d' % (i + 1)) 

# Menampilkan patch langit
for i, patch in enumerate(sky_patches): 
    ax = fig.add_subplot(3, len(sky_patches), len(sky_patches)*2 + i + 1) 
    ax.imshow(patch, cmap=plt.cm.gray, vmin=0, vmax=255) 
    ax.set_xlabel('Langit %d' % (i + 1)) 

# Menampilkan
fig.suptitle('Fitur matriks co-occurrence tingkat abu-abu', fontsize=14, y=1.05) 
plt.tight_layout() 
plt.show()
