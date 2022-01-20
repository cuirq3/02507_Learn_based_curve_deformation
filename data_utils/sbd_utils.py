import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch, cv2, math, os, h5py, tqdm
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes
from torchvision import datasets, transforms

plt.rcParams["savefig.bbox"] = "tight"

transform = transforms.Compose([
    transforms.ToPILImage(),
    # resize
    transforms.Resize((224, 224)),
    # to-tensor
    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def process_mat_inst(file_path):
    mat = scipy.io.loadmat(file_path)
    ann = mat['GTinst'][0].item()[0]
    return ann

def process_mat_seg(file_path):
    mat = scipy.io.loadmat(file_path)
    ann = mat['GTcls'][0].item()[1]
    return ann

def get_bbox(seg, img):
    mask_tensor = torch.tensor(seg).unsqueeze(0)
    img_tensor = torch.tensor(img).permute(2,0,1)
    obj_ids = torch.unique(mask_tensor)
    obj_ids = obj_ids[1:]
    masks = mask_tensor == obj_ids[:,None,None]
    drawn_masks = []
    for mask in masks:
        drawn_masks.append(draw_segmentation_masks(img_tensor, mask, alpha=0.8, colors="blue"))

    # show(drawn_masks)
    boxes = masks_to_boxes(masks)
    drawn_boxes = draw_bounding_boxes(img_tensor, boxes, colors="red")
    # show(drawn_boxes)
    return boxes.numpy(), obj_ids

def get_bbox_center(bbox):
    center = []
    center.append(int((bbox[0]+bbox[2])/2))
    center.append(int((bbox[1]+bbox[3])/2))
    return np.array(center)

def init_poly_vertices(bbox, sample_num, img):
    height, width, _ = img.shape
    center = get_bbox_center(bbox)
    radius = np.array(bbox[:2]) - center
    radius = np.linalg.norm(radius)
    angles = [n*360/sample_num for n in range(sample_num)]
    vertices = []
    for angle in angles:
        vertex = center + radius*np.array([math.sin(angle), math.cos(angle)])
        vertex[0] = max(vertex[0], 0)
        vertex[0] = min(vertex[0], width)
        vertex[1] = min(vertex[1], height)
        vertex[1] = max(vertex[1], 0)
        vertices.append(vertex)
    # draw_poly_vertices(vertices, img)
    return vertices

def get_border_point_on_some_direction(center, angle, radius, seg, height, width, obj_id):
    l_radius = 0
    h_radius = radius
    now_coord = []
    while(h_radius - l_radius > 1):
        now_radius = (l_radius + h_radius)/2
        now_coord = center + now_radius*np.array([math.sin(angle), math.cos(angle)])
        now_coord[0] = max(now_coord[0], 0)
        now_coord[0] = min(now_coord[0], width - 1)
        now_coord[1] = min(now_coord[1], height - 1)
        now_coord[1] = max(now_coord[1], 0)
        if (seg[int(now_coord[1])][int(now_coord[0])] == obj_id):
            l_radius = now_radius + 1
        else:
            h_radius = now_radius - 1
    vertex = now_coord
    return vertex

def gt_poly_vertices(bbox, sample_num, img, seg, obj_id):
    height, width, _ = img.shape
    center = get_bbox_center(bbox)
    radius = np.array(bbox[:2]) - center
    radius = np.linalg.norm(radius)
    angles = [n * 360 / sample_num for n in range(sample_num)]
    vertices = []
    for angle in angles:
        vertices.append(get_border_point_on_some_direction(center, angle, radius, seg, height, width, obj_id))
    # draw_poly_vertices(vertices, img)
    return vertices

def draw_poly_vertices(vertices, img):
    plt.imshow(img)
    for vertex in vertices:
        plt.plot(vertex[0], vertex[1], 'ob', markersize=5)
    plt.show()

def find_extreme_point(bbox, seg, obj_id, img, isDraw):
    extreme_points = []
    center = get_bbox_center(bbox)
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    for i in range(4):
        proposal = []
        if(i%2==0):
            for coord, pixel in enumerate(seg[:,int(bbox[i])]):
                if (pixel == obj_id):
                    proposal.append(coord)
        else:
            for coord, pixel in enumerate(seg[int(bbox[i]),:]):
                if (pixel == obj_id):
                    proposal.append(coord)
        proposal = np.array(proposal)
        if len(proposal) != 0:
            proposal_dis = proposal - center[(i+1)%2]
            proposal_dis = np.abs(proposal_dis)
            extreme_coord = proposal[np.argmax(proposal_dis)]
            if i%2 == 0:
                extreme_points.append([bbox[i], extreme_coord])
            else:
                extreme_points.append([extreme_coord, bbox[i]])
        else:
            if i % 2 == 0:
                extreme_points.append([bbox[i], center[1]+int(bbox_height/4)])
            else:
                extreme_points.append([center[0]+int(bbox_width/4), bbox[i]])
    if isDraw:
        draw_poly_vertices(extreme_points, img)
        plt.show()
    return extreme_points

def organize_hdf5_dataset(root, item_list_name):
    item_list_path = os.path.join(root, item_list_name + '.txt')
    f = open(item_list_path)
    item_list = f.readlines()
    f.close()
    image_root = os.path.join(root, 'img')
    seg_root = os.path.join(root, 'cls')
    border_root = os.path.join(root, 'inst')
    num_sample = 40
    f_list = open(os.path.join(root, 'img_list_' + item_list_name + '.txt'), 'w')

    for line in tqdm.tqdm(item_list):
        file_name = line.split('\n')[0]
        original_img_path = os.path.join(image_root, file_name + '.jpg')
        img = cv2.imread(original_img_path)
        seg = process_mat_seg(os.path.join(seg_root, file_name + '.mat'))
        bboxes, obj_ids = get_bbox(seg, img)
        gt_vertices = []
        init_vertices = []
        img_sizes = []
        imgs = []
        for id, bbox in enumerate(bboxes):
            init_vertices.append(init_poly_vertices(bbox, num_sample, img))
            gt_vertices.append(gt_poly_vertices(bbox, num_sample, img, seg, obj_ids[id]))
            imgs.append(transform(img).numpy())
            width = img.shape[1]
            height = img.shape[0]
            img_sizes.append([width, height])
            f_list.write(original_img_path + '\n')
        imgs = np.array(imgs)
        gt_vertices = np.array(gt_vertices)
        init_vertices = np.array(init_vertices)
        img_sizes = np.array(img_sizes)
        save_file_path = os.path.join(root, item_list_name + '.hdf5')
        if not os.path.exists(save_file_path):
            with h5py.File(save_file_path, 'w') as f:
                f.create_dataset("img", data=imgs, compression="gzip", chunks=True, maxshape=(None,3,224,224))
                f.create_dataset("gt_vert", data=gt_vertices, compression="gzip", chunks=True, maxshape=(None,40,2))
                f.create_dataset("init_vert", data=init_vertices, compression="gzip", chunks=True, maxshape=(None,40,2))
                f.create_dataset("img_size", data=img_sizes, compression="gzip", chunks=True, maxshape=(None,2))
        else:
            with h5py.File(save_file_path, 'a') as f:
                f["img"].resize((f["img"].shape[0] + imgs.shape[0]), axis=0)
                f["img"][-imgs.shape[0]:] = imgs

                f["gt_vert"].resize((f["gt_vert"].shape[0] + gt_vertices.shape[0]), axis=0)
                f["gt_vert"][-gt_vertices.shape[0]:] = gt_vertices

                f["init_vert"].resize((f["init_vert"].shape[0] + init_vertices.shape[0]), axis=0)
                f["init_vert"][-init_vertices.shape[0]:] = init_vertices

                f["img_size"].resize((f["img_size"].shape[0] + img_sizes.shape[0]), axis=0)
                f["img_size"][-img_sizes.shape[0]:] = img_sizes
    f_list.close()

if __name__ == '__main__':
    isTrain = True
    item_list_name = 'train' if isTrain else 'val'
    root = r'E:\DTU\Jan_2022\Deep_snake\snake\data\sbd'
    # organize_hdf5_dataset(root, item_list_name)


    file_name = '2008_000074'
    border = process_mat_inst(os.path.join(root, 'inst', file_name + '.mat'))
    seg = process_mat_seg(os.path.join(root, 'cls', file_name + '.mat'))
    img = cv2.imread(os.path.join(root, 'img', file_name + '.jpg'))
    bboxes, obj_ids = get_bbox(seg, img)
    vertices = []
    for id, bbox in enumerate(bboxes):
        extreme_points = find_extreme_point(bbox, seg, obj_ids[id], img)
        vertices = init_poly_vertices(bbox, 40, img)
        gt_poly_vertices(bbox, 40, img, seg, obj_ids[id])


    plt.imshow(border)
    plt.show()

