import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self,
                 image_size,
                 crop_mode=None,
                 precrop_size=None,
                 image_augmenter=None):
        self.image_size = image_size # width, height
        self.crop_mode = crop_mode
        self.precrop_size = precrop_size if precrop_size is not None else (int(image_size[0]*1.25),
                                                                           int(image_size[1]*1.25))
        self.image_augmenter = image_augmenter

    def preprocess_image(self, im):

        if self.crop_mode == "random":
            im = cv2.resize(im, self.precrop_size)
            row_max_offset = int((im.shape[0] - self.image_size[0]))
            col_max_offset = int((im.shape[1] - self.image_size[1]))
            row_offset = np.random.randint(0, row_max_offset)
            col_offset = np.random.randint(0, col_max_offset)
            im = im[row_offset:row_offset+self.image_size[0], col_offset:col_offset+self.image_size[1], :]
        elif self.crop_mode == "center":
            im = cv2.resize(im, self.precrop_size)
            row_offset = int((im.shape[0] - self.image_size[0])/2)
            col_offset = int((im.shape[1] - self.image_size[1])/2)
            im = im[row_offset:row_offset+self.image_size[0], col_offset:col_offset+self.image_size[1], :]
        else:
            im = cv2.resize(im, self.image_size)

        if self.image_augmenter is not None:
            self.image_augmenter.augment(im)

        im = im.astype(np.float32).transpose(2,0,1)
        im -= 128.0
        
        return im

    def load_image(self, image_path):
        im = cv2.imread(image_path)

        return self.preprocess_image(im)


        
