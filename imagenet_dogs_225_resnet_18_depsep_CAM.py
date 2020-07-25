import cv2, json, os, sys
from tqdm import tqdm
import numpy as np
import cupy as cp
from data_loading.image_data_loader import ImageDataLoader
from data_loading.image_preprocessor import ImagePreprocessor
from imagenet_dogs_225_resnet_18_depsep import ResNet18

BATCH_SIZE = 60
data_folder = "/home/will/Datasets"
im_dir = "./dog_images"
number_of_classes = 120

def forward_to_named_layer(network, X, layer_name):
    for layer in network.layers:
        try:
            X = layer.forward(X, test_mode=True)
            if layer.layer_name == layer_name:
                return X
        except Exception as e:
            print("Error in forward_to_named_layer: {}".format(repr(e)))
    print("Didn't find layer called {}".format(layer_name))
    return None

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 224x224
    size_upsample = (224, 224)
    bz, nc, height, width = feature_conv.shape

    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx, :].dot(feature_conv.reshape((nc, height * width)))
        cam = cam.reshape(h, w)
        cam = cv2.resize(cam, size_upsample)
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)

        if np.max(cam) > 0:
            cam_img = cam / np.max(cam)
        else:
            cam_img = cam
        output_cam.append(cam_img)

    return output_cam

def save_outputs(save_dir, orig_image, output_cam_list, class_name_list):
    orig_image = cv2.resize(orig_image, output_cam_list[0].shape)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for ind, (class_name, cam_array) in enumerate(zip(class_name_list, output_cam_list)):
        cam_array = show_cam_on_image(orig_image, cam_array)
        cv2.imwrite(os.path.join(save_dir, str(ind) + "_" + class_name + ".png"), cam_array)

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    return np.uint8(255 * cam)

val_data_loader = ImageDataLoader(os.path.join(data_folder, "ImageNet2012/ILSVRC2012_dogs/val_img"),
                                 BATCH_SIZE,
                                 image_size=(225,225),
                                 crop_mode="center", start_thread=False)
preprocessor = ImagePreprocessor(image_size=(225,225), crop_mode="center")

experiment_name = "DogsImageNet225ResNet18DepSep"
num_to_dog_name_map_fname = "./imagenet_dog_class_names/num_to_dog_name_map.json"
with open(num_to_dog_name_map_fname, "r") as f:
    num_to_dog_name_map = json.load(f)
network = ResNet18("", load_layers=False)
network.load_network_from_json_and_h5(os.path.join(experiment_name, experiment_name + ".json"),
                                      os.path.join(experiment_name, "epoch_26_testacc_0.686.h5"))

for l in network.layers:
    if l.layer_name == "dense1":
        dense_weights = l.learned_params["weights"].reshape((-1, number_of_classes)).transpose(1, 0)

for im_path in os.listdir(im_dir):
    if not(os.path.isdir(os.path.join(im_dir, im_path))):
        im = val_data_loader.load_image(os.path.join(im_dir, im_path))
        X = im.reshape((1,) + im.shape)
        loss, batch_scores = network.forward(X,
                                            y_one_hot=None,
                                            test_mode=True)
        scores = batch_scores[0,:]
        best = np.argsort(cp.asnumpy(scores))[::-1]

        pre_pooled_data = forward_to_named_layer(network, X, "res8")
        output_cam = returnCAM(cp.asnumpy(pre_pooled_data), dense_weights, best[:3])
        save_outputs(
                "CAM_outputs/" + os.path.splitext(im_path)[0],
                im.transpose([1, 2, 0]) + 128.0,
                output_cam,
                [num_to_dog_name_map[str(b)] for b in best[:3]]
        )

        print("###########################")
        for i in range(5):
            print(im_path, best[i], scores[best[i]], num_to_dog_name_map[str(best[i])])
        plain_im = cv2.imread(os.path.join(im_dir, im_path))
        print(plain_im.shape)
        cv2.putText(plain_im, num_to_dog_name_map[str(best[0])], (int(plain_im.shape[0]/10),int(plain_im.shape[1]/10)),
                    cv2.FONT_HERSHEY_SIMPLEX, min(plain_im.shape[0], plain_im.shape[1])/1000, (0, 255, 100), 5)
        cv2.imwrite(os.path.join(im_dir, "outputs", im_path), plain_im)
