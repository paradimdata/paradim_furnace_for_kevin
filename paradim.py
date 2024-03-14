"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
#import tensorflow as tf

    
    
# Root directory of the project
ROOT_DIR = os.path.abspath("/home/idies/workspace/Storage/ncarey/persistent/PARADIM/furnace_ml/Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

#but first, a hack
import keras.backend

K = keras.backend.backend()
if K=='tensorflow':
    keras.backend.set_image_dim_ordering('tf')
    
    
class ParadimConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "paradim"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 + 1 + 1# Background + goodMelt + fastTop + fastBottom

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5
    
    
    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    
    TRAIN_ROIS_PER_IMAGE = 64
    
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.3
    
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 432
    IMAGE_MAX_DIM = 576



############################################################
#  Dataset
############################################################

class ParadimDataset(utils.Dataset):

    def load_paradim(self, dataset_dir, subset):
        """Load a subset of the crystal dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("paradim", 1, "goodMelt")
        self.add_class("paradim", 2, "fastBottom")
        self.add_class("paradim", 3, "fastTop")
        
        # Train or validation dataset?
        assert subset in ["train", "val", "current"]
        dataset_dir = os.path.join(dataset_dir, subset)

        
        
        for classdir in os.listdir(dataset_dir):
            current_class=0
            if classdir == "goodMelt":
                current_class = 1
            elif classdir == "fastBottom":
                current_class = 2
            elif classdir == "fastTop":
                current_class = 3
            else:
                print("error: dataset class directory not recognized")
                continue
                
            for jsonfile in os.listdir(os.path.join(dataset_dir, classdir)):
                if ".json" in jsonfile:
        
                    annotations = json.load(open(os.path.join(dataset_dir, classdir, jsonfile)))

                    polygons = annotations['shapes'][0]['points']

                    image_path = os.path.join(dataset_dir, classdir, annotations['imagePath'])
                    image = skimage.io.imread(image_path)
                    height, width = image.shape[:2]

                    self.add_image(
                        "paradim",
                        image_id=annotations['imagePath'],  # use file name as a unique image id
                        path=image_path,
                        width=width, height=height,
                        polygons=polygons,
                        class_id = current_class)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        #image_info = self.image_info[image_id]
        #if image_info["source"] != "ballistic":
        #    return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        class_id = info['class_id']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
      
        i = 0
        # Get indexes of pixels inside the polygon and set them to 1
        x = []
        y = []
        for elem in info["polygons"]:
            x.append(elem[0])
            y.append(elem[1])
        rr, cc = skimage.draw.polygon(y, x)
        mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance
        return mask.astype(np.bool), np.full([mask.shape[-1]], class_id, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "paradim":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = ParadimDataset()
    dataset_train.load_paradim(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ParadimDataset()
    dataset_val.load_paradim(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=8,
                layers='heads')

def evaluate(model, dataset_arg, results_file):
    
    # Validation dataset
    dataset_val = ParadimDataset()
    dataset_val.load_paradim(dataset_arg, "val")
    dataset_val.prepare()
    
    for image_id in dataset_val.image_ids:
         # Load image
        image = dataset_val.load_image(image_id)
        gt_class_id = dataset_val.image_info[image_id]['class_id']
        # Run detection
        r = model.detect([image], verbose=0)[0]
        
        max_score = 0
        max_score_index = -1
        for i in range(0, len(r['scores'])):
            if r['scores'][i] > max_score:
                max_score = r['scores'][i]
                max_score_index = i
        detected_class = 0
        if len(r['class_ids']) > 0:
            detected_class = r['class_ids'][max_score_index]
        
        
        #this is where I could calculate intersection over union
        gt_mask, gt_blah = dataset_val.load_mask(image_id)
        mask = r['masks']
        iou = 0
        intersection = 0
        union = 0
        if len(mask) > 0:
            for i in range(0, len(gt_mask)):
                for j in range(0,len(gt_mask[0])):
                    if mask[i][j][max_score_index] and gt_mask[i][j][0]:
                        intersection = intersection + 1
                    if mask[i][j][max_score_index] or gt_mask[i][j][0]:
                        union = union + 1
            iou = intersection / union
        
        with open(results_file, 'a') as results:
            results.write("Score: {0}, IoU: {3}, Class:{1}, GT: {2}\n".format(max_score, detected_class, dataset_val.image_info[image_id]['class_id'], iou))
            
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect crystals.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash' or 'eval")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Ballistic dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')    
    parser.add_argument('--results', required=False,
                        metavar="path to results file to write to",
                        help='resultsFile')
    parser.add_argument('--steps_per_epoch', required=True,
                        metavar="set to 100 by default",
                        help='stepsperpe')
    
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train" or args.command == "eval":
        class ExperimentConfig(ParadimConfig):
            STEPS_PER_EPOCH = int(args.steps_per_epoch)
        config = ExperimentConfig()
    else:
        class InferenceConfig(ParadimConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    
    # Create model
    if args.command == "train":
        #Try forcing CPU training
        #with tf.device('/cpu:0'):
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    elif args.command == "eval":
        evaluate(model, args.dataset, args.results)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
