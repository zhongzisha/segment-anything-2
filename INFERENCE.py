# segment image region using  fine tune model
# See Train.py on how to fine tune/train the model
import sys,os
import numpy as np
import torch
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from matplotlib import pyplot as plt
import utils


# load image
image_path = '/lscratch/{}/pannuke/all_images/1_1790.png'.format(os.environ['SLURM_JOB_ID']) # sys.argv[1]  
mask_path = '/lscratch/{}/pannuke/all_labels/1_1790.npy'.format(os.environ['SLURM_JOB_ID']) # sys.argv[2]  
image_path = '/home/zhongz2/my_llava/examples/figure-002-a71770_large.jpg'
image = cv2.imread(image_path)[...,::-1]  # read image as rgb
r = np.min([1024 / image.shape[1], 1024 / image.shape[0]])
image = cv2.resize(image, (int(image.shape[1] * r), int(image.shape[0] * r)))
height, width = image.shape[:2]
xs = np.arange(50, width-50, 50)
ys = np.arange(50, height-50, 50)
input_points = np.stack(np.meshgrid(xs, ys)).reshape(2, -1).T[:, None, :]
input_labels = np.ones([input_points.shape[0],1])

# use bfloat16 for the entire script (memory efficient)
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()


# Load model you need to have pretrained model already made
sam2_checkpoint = "checkpoints/sam2_hiera_small.pt" # "sam2_hiera_large.pt"
model_cfg = "sam2_hiera_s.yaml" # "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

# Build net and load weights
predictor = SAM2ImagePredictor(sam2_model)
predictor.model.load_state_dict(torch.load("model.torch"))
predictor.set_image(image)
masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=False
)
# sorted_ind = np.argsort(scores)[::-1]
# masks = masks[sorted_ind]
# scores = scores[sorted_ind]
# logits = logits[sorted_ind]
# utils.show_masks(image, masks, scores, point_coords=input_points, box_coords=None, input_labels=input_labels, borders=True, save_root='./')
plt.figure(figsize=(20, 20))
plt.imshow(image)
for mask in masks:
    utils.show_mask(mask.squeeze(0), plt.gca(), random_color=True, borders=False)
# for box in input_boxes:
#     show_box(box, plt.gca())
plt.axis('off')
plt.savefig('./debug_infer_0.png')
plt.close()


plt.figure(figsize=(20, 20))
plt.imshow(image)
# mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=64,
    points_per_batch=128,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.92,
    stability_score_offset=0.7,
    crop_n_layers=1,
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=25.0,
    use_m2m=True,
)
mask_generator.predictor.model.load_state_dict(torch.load("model.torch"))
masks = mask_generator.generate(image)
utils.show_anns(masks, plt.gca(), borders=False)
plt.axis('off')
plt.savefig('debug_infer_1.png')
plt.close()


