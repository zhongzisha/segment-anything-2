# segment image region using  fine tune model
# See Train.py on how to fine tune/train the model
import numpy as np
import torch
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from matplotlib import pyplot as plt


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.8]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)


if __name__ == '__main__':
    # load image
    image_path = '/home/zhongz2/my_llava/examples/figure-002-a71770_large.jpg'
    image = cv2.imread(image_path)[...,::-1]  # read image as rgb
    r = np.min([1024 / image.shape[1], 1024 / image.shape[0]])
    image = cv2.resize(image, (int(image.shape[1] * r), int(image.shape[0] * r)))


    # use bfloat16 for the entire script (memory efficient)
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()


    # Load model you need to have pretrained model already made
    sam2_checkpoint = "checkpoints/sam2_hiera_small.pt" # "sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_s.yaml" # "sam2_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda", apply_postprocessing=False)

    # Build net and load weights
    # predictor = SAM2ImagePredictor(sam2_model)
    # predictor.model.load_state_dict(torch.load("model.torch"))
    mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
    mask_generator.predictor.model.load_state_dict(torch.load("model.torch"))
    masks = mask_generator.generate(image)
    fig = plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig('/data/zhongz2/temp29/debug_infer.png')
    plt.close()

    mask_generator_2 = SAM2AutomaticMaskGenerator(
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
    mask_generator_2.predictor.model.load_state_dict(torch.load("model.torch"))
    masks = mask_generator.generate(image)
    fig = plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig('/data/zhongz2/temp29/debug_infer_2.png')
    plt.close()


