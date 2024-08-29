





import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
import utils
import json
from pycocotools import mask as maskUtils
import base64


def process_fold(fold, input_path, output_path) -> None:
    fold_path = Path(input_path) / f"fold{fold}"
    output_fold_path = Path(output_path) / f"fold{fold}"
    output_fold_path.mkdir(exist_ok=True, parents=True)
    (output_fold_path / "images").mkdir(exist_ok=True, parents=True)
    (output_fold_path / "labels").mkdir(exist_ok=True, parents=True)

    print(f"Fold: {fold}")
    print("Loading large numpy files, this may take a while")
    images = np.load(fold_path / "images.npy")
    masks = np.load(fold_path / "masks.npy")

    print("Process images")
    images_list = []
    for i in tqdm(range(len(images)), total=len(images)):
        outname = f"{fold}_{i}.png"
        out_img = images[i]
        im = Image.fromarray(out_img.astype(np.uint8))
        width, height = im.size
        im.save(output_fold_path / "images" / outname)
        images_list.append({
            "id": i+1,
            "file_name": outname,
            "height": height,
            "width": width
        })

    print("Process masks")
    labels_dict = {1: 'Neoplastic', 2: 'Inflammatory', 3: 'Connective', 4: 'Dead', 5: 'Epithelial', 0: 'Background'}
    annotations = []
    annotation_id = 1
    for i in tqdm(range(len(images)), total=len(images)):
        outname = f"{fold}_{i}.npy"

        # need to create instance map and type map with shape 256x256
        mask = masks[i]
        inst_map = np.zeros((256, 256))
        num_nuc = 0
        for j in range(5):
            # copy value from new array if value is not equal 0
            layer_res = utils.remap_label(mask[:, :, j])
            # inst_map = np.where(mask[:,:,j] != 0, mask[:,:,j], inst_map)
            inst_map = np.where(layer_res != 0, layer_res + num_nuc, inst_map)
            num_nuc = num_nuc + np.max(layer_res)
        inst_map = utils.remap_label(inst_map)

        type_map = np.zeros((256, 256)).astype(np.int32)
        for j in range(5):
            layer_res = ((j + 1) * np.clip(mask[:, :, j], 0, 1)).astype(np.int32)
            type_map = np.where(layer_res != 0, layer_res, type_map)

        outdict = {"inst_map": inst_map, "type_map": type_map}
        np.save(output_fold_path / "labels" / outname, outdict)


        for j in range(5): # ignore background
            instance_mask = mask[:, :, j]
            
            for label in np.unique(instance_mask)[1:]:
                instance_mask1 = (instance_mask == label).astype(np.uint8)
                # Find contours
                contours, _ = cv2.findContours(instance_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Create RLE for segmentation
                rle = maskUtils.encode(np.asfortranarray(instance_mask1.astype(np.uint8)))
                area = int(maskUtils.area(rle))
                x, y, w, h = maskUtils.toBbox(rle)
                # Bounding box
                # x, y, w, h = cv2.boundingRect(contours[0])
                rle['counts'] = base64.b64encode(rle['counts']).decode('utf-8')
                annotations.append({
                    "id": annotation_id,
                    "image_id": i+1,
                    "category_id": 0 if j==5 else j+1,
                    "segmentation": rle,
                    "area": area,
                    "bbox": [x, y, w, h],
                    "iscrowd": 0
                })
                
                annotation_id += 1

    categories = [
        {
            "id": label,
            "name": label_name,
            "supercategory": "cell"
        }
        for label, label_name in labels_dict.items()
    ]

    coco_dict = {
        "images": images_list,
        "annotations": annotations,
        "categories": categories
    }

    # Step 7: Save as JSON
    with open(os.path.join(output_fold_path, 'instances.json'), 'w') as json_file:
        json.dump(coco_dict, json_file)


# process_fold(1, '/data/zhongz2/data/PanNuke', '/lscratch/'+os.environ['SLURM_JOB_ID']+'/pannuke')
# process_fold(2, '/data/zhongz2/data/PanNuke', '/lscratch/'+os.environ['SLURM_JOB_ID']+'/pannuke')
# process_fold(3, '/data/zhongz2/data/PanNuke', '/lscratch/'+os.environ['SLURM_JOB_ID']+'/pannuke')
def debug():
    image_path = '/lscratch/34740217/pannuke/fold1/images/1_332.png'
    mask_path = '/lscratch/34740217/pannuke/fold1/labels/1_332.npy'
    image = cv2.imread(image_path)
    masks = np.load(mask_path, allow_pickle=True)
    inst_map = masks[()]["inst_map"].astype(np.int32)
    type_map = masks[()]["type_map"].astype(np.int32)

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image)
    utils.show_mask(inst_map, plt.gca(), random_color=True, borders=False)
    plt.axis('off')
    plt.savefig('/data/zhongz2/temp29/cell_instance_seg_mask.png')
    plt.close()


    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image)
    utils.show_mask(type_map, plt.gca(), random_color=True, borders=False)
    plt.axis('off')
    plt.savefig('/data/zhongz2/temp29/cell_type_seg_mask.png')
    plt.close()






