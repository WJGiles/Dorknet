import cv2, json, os, sys
from tqdm import tqdm
import numpy as np
import cupy as cp
from data_loading.image_data_loader import ImageDataLoader
from imagenet_dogs_225_resnet_18_depsep import ResNet18

BATCH_SIZE = 60
data_folder = "/home/will/Datasets"
im_dir = "./dog_images"

val_data_loader = ImageDataLoader(os.path.join(data_folder, "ImageNet2012/ILSVRC2012_dogs/val_img"),
                                 BATCH_SIZE,
                                 image_size=(225,225),
                                 crop_mode="center", start_thread=True)

experiment_name = "DogsImageNet225ResNet18DepSep"
num_to_dog_name_map_fname = "./imagenet_dog_class_names/num_to_dog_name_map.json"
with open(num_to_dog_name_map_fname, "r") as f:
    num_to_dog_name_map = json.load(f)
network = ResNet18("", load_layers=False)
network.load_network_from_json_and_h5(os.path.join(experiment_name, experiment_name + ".json"),
                                      os.path.join(experiment_name, "epoch_26_testacc_0.686.h5"))
print("Testing...")
test_acc = network.test(val_data_loader.pull_batch(int(120*50/BATCH_SIZE)), BATCH_SIZE, 120*50)
val_data_loader.stop_thread()
print("Test acc: {}".format(test_acc))

for im_path in os.listdir(im_dir):
    if not(os.path.isdir(os.path.join(im_dir, im_path))):
        im = val_data_loader.load_image(os.path.join(im_dir, im_path))
        X = im.reshape((1,) + im.shape)
        loss, batch_scores = network.forward(X, y_one_hot=None, test_mode=True)
        scores = batch_scores[0,:]
        best = np.argsort(cp.asnumpy(scores))[::-1]
        print("###########################")
        for i in range(5):
            print(im_path, best[i], scores[best[i]], num_to_dog_name_map[str(best[i])])
        plain_im = cv2.imread(os.path.join(im_dir, im_path))
        cv2.putText(plain_im, num_to_dog_name_map[str(best[0])], (int(plain_im.shape[0]/10),int(plain_im.shape[1]/10)),
                    cv2.FONT_HERSHEY_SIMPLEX, min(plain_im.shape[0], plain_im.shape[1])/1000, (0, 255, 100), 5)
        cv2.imwrite(os.path.join(im_dir, "outputs", im_path), plain_im)


