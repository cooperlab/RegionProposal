import os
import openslide
import cv2
import numpy as np

def convert_images(path):

    samples = []
    for fn in os.listdir(path):
        if '.svs' in fn:
            slide = openslide.OpenSlide(path + fn)
            k = slide.level_count - 2
            dim = slide.level_dimensions[k]
            img = np.array(slide.read_region((0,0), k, dim))

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            final_img = cv2.resize(gray_img, (500, 500))
            cv2.imshow('teste', final_img)
            cv2.waitKey(0)
    return samples

def main():
    #path_folder_1 = '/home/lcoop22/Images/BRCA/'
    #path_folder_2 = '/home/lcoop22/Images/LGG/'
    path_folder_1 = '/home/nelson/LGG-test/'
    convert_images(path_folder_1)
    return

if __name__ == "__main__":
    main()


