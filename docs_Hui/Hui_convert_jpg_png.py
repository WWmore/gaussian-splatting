
from PIL import Image
from pillow_heif import register_heif_opener

import os

path = os.path.dirname(os.path.abspath(__file__))
#print(path) ## c:\Users\WANGH0M\gaussian-splatting

# Get list of HEIF and HEIC files in directory
#directory = path + '\data_bonsai_old\input'

directory = r'C:\Users\WANGH0M\gaussian-splatting\data_bonsai_old\input'
directory = r'C:\Users\WANGH0M\gaussian-splatting\data_bonsai_new\input'
directory = r'C:\Users\WANGH0M\gaussian-splatting\data_office\input'

files = [f for f in os.listdir(directory) if f.endswith('.HEIC') or f.endswith('.heif')]
#print(files)

# Convert each file to JPEG
for filename in files:
    #image = Image.open(os.path.join(directory, filename))

    register_heif_opener()
    image = Image.open(directory + '\\' + filename)
    # print(image)

    half = 0.5
    out = image.resize( [int(half * s) for s in image.size] )

    out.save(os.path.join(directory, os.path.splitext(filename)[0] + '.png'))
