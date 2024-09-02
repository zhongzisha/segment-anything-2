





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
import pdb

def process_fold(fold, input_path, output_path, image_id_start=0, annotation_id_start=0) -> None:
    fold_path = Path(input_path) / f"fold{fold}"
    output_fold_path = Path(output_path) # / f"fold{fold}"
    output_fold_path.mkdir(exist_ok=True, parents=True)
    (output_fold_path / "all_images").mkdir(exist_ok=True, parents=True)
    (output_fold_path / "all_labels").mkdir(exist_ok=True, parents=True)

    print(f"Fold: {fold}")
    print("Loading large numpy files, this may take a while")
    images = np.load(fold_path / "images.npy")
    masks = np.load(fold_path / "masks.npy")

    print("Process images")
    images_list = []
    for i in tqdm(range(len(images)), total=len(images)):
        image_id = image_id_start+i
        outname = f"{fold}_{i}.png"
        out_img = images[i]
        im = Image.fromarray(out_img.astype(np.uint8))
        width, height = im.size
        im.save(output_fold_path / "all_images" / outname)
        images_list.append({
            "id": image_id,
            "file_name": outname,
            "height": height,
            "width": width
        })

    print("Process masks")
    annotations = []
    annotation_id = annotation_id_start
    for i in tqdm(range(len(images)), total=len(images)):
        image_id = image_id_start+i
        outname = f"{fold}_{i}.npy"

        # need to create instance map and type map with shape 256x256
        mask = masks[i]
        H, W = mask.shape[:2]
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
        np.save(output_fold_path / "all_labels" / outname, outdict)


        for j in range(5): # ignore background
            instance_mask = mask[:, :, j]
            
            for label in np.unique(instance_mask)[1:]:
                instance_mask1 = (instance_mask == label).astype(np.uint8)
                # Find contours
                contours, _ = cv2.findContours(instance_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=lambda x: cv2.contourArea(x))
                area = cv2.contourArea(contours[0])
                # Create RLE for segmentation
                # rle = maskUtils.encode(np.asfortranarray(instance_mask1.astype(np.uint8)))
                # area = int(maskUtils.area(rle))
                # x, y, w, h = maskUtils.toBbox(rle)
                x, y, w, h = cv2.boundingRect(contours[0])
                # x, y = max(0, x), max(0, y)
                # if (x+w) >= W: 
                #     w = W - x
                # if (y+h) >= H:
                #     h = H - y
                # Bounding box
                # x, y, w, h = cv2.boundingRect(contours[0])
                # rle['counts'] = base64.b64encode(rle['counts']).decode('utf-8')
                poly = contours[0].flatten().tolist()
                if len(poly) < 6:
                    continue
                # for contour in contours:
                #     # Flatten the contour to get a list of x, y coordinates
                #     contour = contour.flatten().tolist()
                #     polygons.append(contour)
                #     break
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 0 if j==5 else j+1,
                    "segmentation": [poly],
                    "area": area,
                    "bbox": [x, y, w, h],
                    "iscrowd": 0
                })
                
                annotation_id += 1
    return images_list, annotations, image_id_start+len(images), annotation_id


def main1():
    output_fold_path = '/lscratch/'+os.environ['SLURM_JOB_ID']+'/pannuke'
    images_list = []
    annotations = []
    image_id_start = 0
    annotation_id_start = 0
    for fold in [1, 2, 3]:
        images_list_, annotations_, image_id_start, annotation_id_start = \
            process_fold(fold, '/data/zhongz2/data/PanNuke', output_fold_path, \
            image_id_start=image_id_start, annotation_id_start=annotation_id_start)
        images_list.extend(images_list_)
        annotations.extend(annotations_)

    labels_dict = {1: 'Neoplastic', 2: 'Inflammatory', 3: 'Connective', 4: 'Dead', 5: 'Epithelial'} # , 0: 'Background'}

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
    image_path = '/lscratch/34912434/pannuke/fold1/all_images/1_0.png'
    mask_path = '/lscratch/34912434/pannuke/fold1/all_labels/1_0.npy'
    image = cv2.imread(image_path)
    masks = np.load(mask_path, allow_pickle=True)
    inst_map = masks[()]["inst_map"].astype(np.int32)
    type_map = masks[()]["type_map"].astype(np.int32)

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image)
    utils.show_mask(inst_map, plt.gca(), random_color=True, borders=False)
    plt.axis('off')
    plt.savefig('./cell_instance_seg_mask.jpg')
    plt.close()


    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image)
    utils.show_mask(type_map, plt.gca(), random_color=True, borders=False)
    plt.axis('off')
    plt.savefig('./cell_type_seg_mask.jpg')
    plt.close()




def visualize_coco():
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from pycocotools.coco import COCO
    from PIL import Image
    import numpy as np
    import os

    def visualize_coco_image(coco, image_id, image_dir, save_dir='./'):
        # Load the image information
        img_info = coco.loadImgs(image_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])

        # Load the image
        image = Image.open(img_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')

        # Load the annotations
        annotation_ids = coco.getAnnIds(imgIds=img_info['id'])
        annotations = coco.loadAnns(annotation_ids)

        # Create a matplotlib axes
        ax = plt.gca()

        for annotation in annotations:
            # Get the segmentation mask
            mask = coco.annToMask(annotation)

            # Display the mask
            masked_image = np.ma.masked_where(mask == 0, mask)
            ax.imshow(masked_image, alpha=0.5, cmap='jet')

            # Get the bounding box coordinates
            bbox = annotation['bbox']
            x, y, width, height = bbox

            # Create a rectangle patch for the bounding box
            rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='yellow', facecolor='none')
            ax.add_patch(rect)

            # Display the category label
            category_id = annotation['category_id']
            category_name = coco.loadCats(category_id)[0]['name']
            plt.text(x, y, category_name, color='white', fontsize=12, backgroundcolor='none')

        # plt.show()
        plt.savefig(os.path.join(save_dir, f'{image_id}.jpg'))
        plt.close()

    # Load the COCO dataset
    coco = COCO('/lscratch/34912434/pannuke/fold1/instances.json')
    image_dir = '/lscratch/34912434/pannuke/fold1/images/'
    # Example usage
    image_id = coco.getImgIds()[0]  # Get the first image ID
    visualize_coco_image(coco, image_id, image_dir)


def create_train_val_splits():
    import json
    import random
    from sklearn.model_selection import train_test_split

    root = '/lscratch/{}/pannuke/'.format(os.environ['SLURM_JOB_ID'])
    annos_dir = '{}/annotations'.format(root)
    os.makedirs(annos_dir, exist_ok=True)

    # Load the COCO-style JSON file
    with open(f'{root}/instances.json', 'r') as f:
        coco_data = json.load(f)

    # Extract images and annotations
    images = coco_data['images']
    annotations = coco_data['annotations']

    # Get all image IDs
    image_ids = [img['id'] for img in images]

    # Split image IDs into train and test
    train_ids, test_ids = train_test_split(image_ids, test_size=0.2, random_state=42)

    # Create train and test splits
    train_images = [img for img in images if img['id'] in train_ids]
    test_images = [img for img in images if img['id'] in test_ids]

    def mklinks(subset, images_list):
        save_dir = os.path.join(root, 'images', subset)
        os.makedirs(save_dir, exist_ok=True)
        for item in images_list:
            src = os.path.join(root, 'all_images', item['file_name'])
            os.system('ln -sf "{}" "{}/"'.format(src, save_dir))
    mklinks('train', train_images)
    mklinks('test', test_images)

    train_annotations = [ann for ann in annotations if ann['image_id'] in train_ids]
    test_annotations = [ann for ann in annotations if ann['image_id'] in test_ids]

    # Keep the categories unchanged
    categories = coco_data['categories']

    # Create new JSON data for train and test
    train_data = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': categories
    }

    test_data = {
        'images': test_images,
        'annotations': test_annotations,
        'categories': categories
    }

    # Save train and test JSON files
    with open(f'{annos_dir}/instances_train.json', 'w') as f:
        json.dump(train_data, f)

    with open(f'{annos_dir}/instances_test.json', 'w') as f:
        json.dump(test_data, f)

    print("Splitting completed successfully!")





def min_index(arr1, arr2):
    """
    Find a pair of indexes with the shortest distance.

    Args:
        arr1: (N, 2).
        arr2: (M, 2).

    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """
    Merge multi segments to one list. Find the coordinates with min distance between each segment, then connect these
    coordinates with one thin line to merge all segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...],
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s



def convert_coco_to_yolo():
    from pathlib import Path
    import json
    from collections import defaultdict
    from tqdm import tqdm
    import numpy as np

    json_dir = '/lscratch/{}/pannuke/annotations'.format(os.environ['SLURM_JOB_ID'])
    save_dir = '/lscratch/{}/pannuke/'.format(os.environ['SLURM_JOB_ID'])
    use_segments = True

    for json_file in sorted(Path(json_dir).resolve().glob("*.json")):
        fn = Path(save_dir) / "labels" / json_file.stem.replace("instances_", "")  # folder name
        fn.mkdir(parents=True, exist_ok=True)
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {"%g" % x["id"]: x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        # Write labels file
        for img_id, anns in tqdm(imgToAnns.items(), desc=f"Annotations {json_file}"):
            img = images["%g" % img_id]
            h, w, f = img["height"], img["width"], img["file_name"]

            bboxes = []
            segments = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls_label = ann["category_id"] - 1  # class
                box = [cls_label] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                # Segments
                if use_segments:
                    if len(ann["segmentation"]) > 1:
                        s = merge_multi_segment(ann["segmentation"])
                        s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                    else:
                        s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                        s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                    s = [cls_label] + s
                    if s not in segments:
                        segments.append(s)

            # Write
            with open((fn / f).with_suffix(".txt"), "a") as file:
                for i in range(len(bboxes)):
                    line = (*(segments[i] if use_segments else bboxes[i]),)  # cls_label, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")


if __name__ == '__main__':
    main1()
    create_train_val_splits()
    convert_coco_to_yolo()
