
##pip install rembg
##pip install --ignore-installed Pillow==9.3.0
"https://github.com/danielgatis/rembg"

import PIL
PIL.__version__
from PIL import Image
import rembg
from pathlib import Path
from rembg import remove, new_session

def remove_1_image(input_path, output_path):
    # input_path = 'input.png'
    # output_path = 'output.png'
    input = Image.open(input_path)
    output = remove(input)
    output.save(output_path)

def remove_batches(folder):
    session = new_session()
    for file in Path(folder).glob('*.png'):
        input_path = str(file)
        output_path = str(file.parent / (file.stem + ".png"))
        output_path = str(file.parent) +'/rmbg/' + (file.stem + ".png")

        with open(input_path, 'rb') as i:
            with open(output_path, 'wb') as o:
                input = i.read()
                output = remove(input, session=session)
                o.write(output)



folder = r'C:\Users\WANGH0M\gaussian-splatting\data_coral_rmbg\images'

#remove_1_image(folder+'\IMG_6115.png',folder+'\IMG_6115_1.png')

remove_batches(folder)





import shutil
"change a batch of images (.png) from the name 'IMG_X_out.png' to the name 'IMG_X.png'. "
# folder + '/IMG_6115'
# shutil.move(name + '_out.png', name)

# for file in Path(folder).glob('*.png'):
#     name = file.stem
#     print(name, name[:-4])
#     shutil.move(folder + '/' + name + '.png', folder + '/' + name[:-4] + '.png')