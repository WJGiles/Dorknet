# Dorknet

Artisanal CNNs. 

* Enough building blocks to put together a lot of serious architectures (ResNets, MobileNets etc). 
* GPU support via Cupy (currently mandatory but work in progress to make this optional - CPU-only version uses Cython for critical parts to improve performance)
* Image data augmentation routines, l2 regularisation and 'mixup'
* Class activation maps (CAM) for visualising image regions responsible for predictions
* Work in progress!

## Installation
 
* Python Virtual Environment - you might need to change the version of cupy to match your version of Cuda and/or driver setup. Please check out the cupy installation guides for this, it's pleasingly straightforward.

  `virtualenv -p python3 env`

  `source env/bin/activate`

  `pip install -r requirements.txt`

* Docker
   * The Docker directory in this repo provides scripts for building and running Dorknet images based on an Intel Python distribution image (which I've found offers a really useful speedup when running in CPU only mode), and an official cupy container. You'll need to adjust paths in volume mappings in the the 'run' scripts to make your datasets visible within the container when running these.

To get started with the obligatory MNIST example, try

   `bash scripts/download_MNIST.sh`
   
   `python -m examples.MNIST_basic_convnet`

There is a release with a zip file containing a trained model - the one used in generating the CAM images below. It's a slight variation on a ResNet18 with depthwise separable convolution layers.

If you've got the model mentioned above and unzipped it in your Dorknet directory, you can have a go with;

   `python -m examples.imagenet_dogs_resnet_18_depsep_CAM`

## Use

Your best bet is to look at the examples and extrapolate.

### Some Class Activation Map outputs

These are produced using a network trained on the 120 dog classes of ImageNet.

![alt text](./CAM_images/0_Bouvier_des_Flandres.png "CAM visualisation")

Region responsible for class 'Bouvier des Flandres'

![alt text](./CAM_images/2_boxer.png "CAM visualisation")

Region responsible for class 'Boxer'

## Documentation 

Nothing official, but I'm working on adding more in the source code where appropriate.