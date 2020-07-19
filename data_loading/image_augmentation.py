import cv2, os
import numpy as np

class ImageAugmenter:

    def __init__(self,
                 hsv_pert_tuples=None,
                 rotation_tuple=None,
                 horizontal_flip_prob=None,
                 translation_tuple=None):
        self.hsv_pert_tuples = hsv_pert_tuples
        self.rotation_tuple = rotation_tuple
        self.translation_tuple = translation_tuple
        self.horizontal_flip_prob = horizontal_flip_prob

    def augment(self, im):
        if self.hsv_pert_tuples is not None:
            hue_pert = np.random.uniform(*self.hsv_pert_tuples[0])
            sat_pert = np.random.uniform(*self.hsv_pert_tuples[1])
            val_pert = np.random.uniform(*self.hsv_pert_tuples[2])

            im = self.hsv_perturbation(im, [hue_pert, sat_pert, val_pert])
        
        if self.rotation_tuple is not None:
            rot_degrees = np.random.uniform(*self.rotation_tuple)
            im = self.rotate_image(im, rot_degrees)

        if self.translation_tuple is not None:
            row_trans = np.random.random_integers(low=-1*self.translation_tuple[0],
                                                  high=self.translation_tuple[1])
            col_trans = np.random.random_integers(low=-1*self.translation_tuple[0],
                                                  high=self.translation_tuple[1])

            im = self.translate_image(im, row_trans, col_trans)

        if self.horizontal_flip_prob is not None:
            im = self.horizontal_flip_image(im)

        return im

    def hsv_perturbation(self, im, pert_proportion):
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32) # recast to avoid overflow
        hsv[:,:,0] *= pert_proportion[0]
        hsv[:,:,1] *= pert_proportion[1]
        hsv[:,:,2] *= pert_proportion[2]
        np.clip(hsv, 0, 255, out=hsv)
        np.clip(hsv[:,:,0], 0, 179, out=hsv[:,:,0])
        im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return im

    def translate_image(self, im, row_trans, col_trans):
        # In theory not that useful because in theory convs have 
        # translation equivariance, but maybe worth a try, there are 
        # still boundary effects etc...
        M = np.float32([[1,0,row_trans],[0,1,col_trans]])
        im = cv2.warpAffine(im, M, dsize=(im.shape[1], im.shape[0]))

        return im

    def rotate_image(self, im, rot_degrees):
        M = cv2.getRotationMatrix2D((im.shape[1]/2,im.shape[0]/2),
                                     rot_degrees,1)
        im = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]))

        return im

    def horizontal_flip_image(self, im):
        if np.random.uniform() < self.horizontal_flip_prob:
            im = im[:,::-1,:]

        return im


if __name__ == "__main__":
    hsv_perts = [(0.9,1.1), (0.5,1.5), (0.5,1.5)]
    translation_tuple = None
    rotation_tuple = (-10, 10)
    augmenter = ImageAugmenter(hsv_pert_tuples=hsv_perts,
                               rotation_tuple=rotation_tuple,
                               translation_tuple=translation_tuple,
                               horizontal_flip_prob=0.5)
    for e, i in enumerate(range(35)):
        #im_path = "/home/will/Datasets/ImageNet2012/ILSVRC2012_dogs/train/n02085620/n02085620_7.JPEG"
        im_path = "/home/will/Datasets/ImageNet2012/ILSVRC2012_dogs/train/n02085620/n02085620_286.JPEG"
        im = cv2.imread(im_path)
        perturbed = augmenter.augment(im)
        #perturbed_out = "{}_hsv_perturbed_{}_{}_{}.jpg".format(
        #    os.path.split(im_path)[-1].split(".")[0], pert_proportion[0], pert_proportion[1], pert_proportion[2]
        #    )
        perturbed_out = "{}.jpg".format(e)
        cv2.imwrite(perturbed_out, perturbed)
