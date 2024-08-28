# Train/Fine-Tune SAM 2 on the LabPics 1 dataset

# Toturial: https://medium.com/@sagieppel/train-fine-tune-segment-anything-2-sam-2-in-60-lines-of-code-928dd29a63b3
# Main repo: https://github.com/facebookresearch/segment-anything-2
# Labpics Dataset can be downloaded from: https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1
# Pretrained models for sam2 Can be downloaded from: https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#download-checkpoints

import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Read data
data_dir="/lscratch/34740217/pannuke/fold1"
data=[] # list of files in dataset
for ff, name in enumerate(os.listdir(data_dir+"/images/")):  # go over all folder annotation
    data.append({"image":data_dir+"/images/"+name,"annotation":data_dir+"/labels/"+name[:-4]+".npy"})

def read_batch(data): # read random image and its annotaion from  the dataset (LabPics)
    ent  = data[np.random.randint(len(data))] # choose random entry
    img_prefix = os.path.splitext(os.path.basename(ent["image"]))[0]
    Img = cv2.imread(ent["image"])[...,::-1]  # read image
    masks = np.load(ent["annotation"], allow_pickle=True)
    inst_map = masks[()]["inst_map"].astype(np.int32)
    type_map = masks[()]["type_map"].astype(np.int32)

   # resize image

    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]]) # scalling factor
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    inst_map = cv2.resize(inst_map, (int(inst_map.shape[1] * r), int(inst_map.shape[0] * r)),interpolation=cv2.INTER_NEAREST)
    type_map = cv2.resize(type_map, (int(type_map.shape[1] * r), int(type_map.shape[0] * r)),interpolation=cv2.INTER_NEAREST)

   # Get binary masks and points

    inds = np.unique(inst_map)[1:] # load all indices
    points= []
    masks = []
    labels = []
    for ind in inds:
        mask=(inst_map == ind).astype(np.uint8) # make binary mask corresponding to index ind
        masks.append(mask)
        coords = np.argwhere(mask > 0) # get all coordinates in mask
        yx = np.array(coords[np.random.randint(len(coords))]) # choose random point/coordinate
        points.append([[yx[1], yx[0]]])
        labels.append(type_map[yx[0], yx[1]])
    return Img,np.array(masks),np.array(points), np.array(labels).reshape(-1, 1),img_prefix

# Load model

sam2_checkpoint = "checkpoints/sam2_hiera_small.pt" # path to model weight (pre model loaded from: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt)
model_cfg = "sam2_hiera_s.yaml" #  model config
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda") # load model
predictor = SAM2ImagePredictor(sam2_model)

# Set training parameters

predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder
predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder
optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=1e-5,weight_decay=4e-5)
scaler = torch.cuda.amp.GradScaler() # mixed precision

# Training loop

for itr in range(100000):
    with torch.cuda.amp.autocast(): # cast to mix precision
        image,mask,input_point, input_label,img_prefix = read_batch(data) # load data batch
        if mask.shape[0]==0: continue # ignore empty batches
        predictor.set_image(image) # apply SAM image encoder to the image

        # prompt encoding

        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),boxes=None,masks=None,)

        # mask decoder

        batched_mode = unnorm_coords.shape[0] > 1 # multi object prediction
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=batched_mode,high_res_features=high_res_features,)
        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])# Upscale the masks to the original image resolution

        # Segmentaion Loss caclulation

        gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
        prd_mask = torch.sigmoid(prd_masks[:, 0])# Turn logit map to probability map
        seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # cross entropy loss

        # Score loss calculation (intersection over union) IOU

        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss=seg_loss+score_loss*0.05  # mix losses

        # apply back propogation

        predictor.model.zero_grad() # empty gradient
        scaler.scale(loss).backward()  # Backpropogate
        scaler.step(optimizer)
        scaler.update() # Mix precision

        if itr%1000==0: torch.save(predictor.model.state_dict(), "model.torch");print("save model")

        # Display results

        if itr==0: mean_iou=0
        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
        print("step)",itr, "Accuracy(IOU)=",mean_iou)
