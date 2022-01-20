from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes
from torchvision import datasets, transforms
import cv2, os, torch, tqdm, h5py
import matplotlib.pyplot as plt
from sbd_utils import show, draw_poly_vertices, get_bbox_center
import numpy as np
from scipy.interpolate import interp1d
import easy_snake as es
import PIL.Image
from argparse import ArgumentParser

transform = transforms.Compose([
    transforms.ToPILImage(),
    # resize
    transforms.Resize((224, 224)),
    # to-tensor
    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def read_annotation(file_path):
    seg = np.array(PIL.Image.open(file_path))
    return seg

def get_bbox(seg, img, isDraw):
    mask_tensor = torch.tensor(seg).unsqueeze(0)
    img_tensor = torch.tensor(img).permute(2,0,1)
    obj_id = torch.tensor([1,1,1])
    mask = (mask_tensor == obj_id[:,None,None])[0]
    drawn_masks = []
    drawn_masks.append(draw_segmentation_masks(img_tensor, mask, alpha=0.8, colors="blue"))

    # show(drawn_masks)
    boxes = masks_to_boxes(mask.unsqueeze(0))
    drawn_boxes = draw_bounding_boxes(img_tensor, boxes, colors="red")
    if isDraw:
        show(drawn_boxes)
        plt.show()
    return boxes.numpy()


def find_extreme_point(bbox, seg, obj_id, img, isDraw=False):
    # Following the order of ymax, xmin, ymin, xmax
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
    extreme_points = [extreme_points[-1]] + extreme_points[:3]
    if isDraw:
        draw_poly_vertices(extreme_points, img)
        plt.show()
    return extreme_points

def get_octagon_points(extreme_points, bbox, img, isDraw=False):
    octagon_points = []
    extreme_points = np.array(extreme_points)
    segment_length = []
    for id, point in enumerate(extreme_points):
        if np.isnan(np.linalg.norm(point - extreme_points[(id+1)%4])):
            print('warning! nan found')
            segment_length.append(0)
            continue
        segment_length.append(np.linalg.norm(point - extreme_points[(id+1)%4]))
    segment_length = np.array(segment_length)
    segment_length /= 4

    # segment 0
    offset = np.array([[0, segment_length[0]], [-segment_length[0], 0]])
    octagon_point = extreme_points[0] + offset[1]
    octagon_points.append([max(octagon_point[0], bbox[0]), octagon_point[1]])
    octagon_point = extreme_points[1] + offset[0]
    octagon_points.append([octagon_point[0], min(octagon_point[1], bbox[3])])

    #segment 0
    offset = np.array([[-segment_length[1], 0], [0, -segment_length[1]]])
    octagon_point = extreme_points[1] + offset[1]
    octagon_points.append([max(octagon_point[0], bbox[0]), octagon_point[1]])
    octagon_point = extreme_points[2] + offset[0]
    octagon_points.append([octagon_point[0], max(octagon_point[1], bbox[1])])


    # segment 1
    offset = np.array([[0, -segment_length[2]], [segment_length[2], 0]])
    octagon_point = extreme_points[2] + offset[1]
    octagon_points.append([min(octagon_point[0], bbox[2]), octagon_point[1]])
    octagon_point = extreme_points[3] + offset[0]
    octagon_points.append([octagon_point[0], max(octagon_point[1], bbox[1])])


    # segment 2
    offset = np.array([[segment_length[3], 0], [0, segment_length[3]]])
    octagon_point = extreme_points[3] + offset[1]
    octagon_points.append([octagon_point[0], min(octagon_point[1], bbox[3])])
    octagon_point = extreme_points[0] + offset[0]
    octagon_points.append([min(octagon_point[0], bbox[2]), octagon_point[1]])

    if isDraw:
        draw_poly_vertices(octagon_points, img)
        plt.show()
    return np.array(octagon_points)

def draw_poly_segments(vertices, img):
    vertices = np.concatenate([vertices, [vertices[0]]], axis=0)
    x_values = vertices[:,0]
    y_values = vertices[:,1]
    plt.plot(x_values, y_values, color='blue', linewidth=2)
    # for id, start_point in enumerate(vertices):
    #     x_values = [start_point[0], end_vertices[id][0]]
    #     y_values = [start_point[1], end_vertices[id][1]]
    #     plt.plot(x_values, y_values, color='blue', linewidth=2)
    plt.imshow(img)
    plt.show()

def get_init_vertices(extreme_points, octagon_points, img, isDraw):
    _, idx = np.unique(octagon_points, return_index=True, axis=0)
    octagon_points = octagon_points[np.sort(idx)]
    vertices = np.concatenate([octagon_points, [octagon_points[0]]], axis=0)
    snake = np.array([vertices[:,0], vertices[:,1]])
    init_vert = es.distribute_points(snake, 128)
    init_vert = init_vert.T
    # adjust order to match ground truth points
    dist = np.linalg.norm(np.array(init_vert - np.array(extreme_points[0])), axis=1)
    start_id = np.argmin(dist)
    init_vert = np.concatenate([init_vert[start_id:], init_vert[:start_id]], axis=0)
    if isDraw:
        draw_poly_vertices(init_vert, img)
        plt.show()
    return init_vert

def get_gt_vertices(extreme_points, img, seg, isDraw):
    mask = seg == 1
    contour = es.largest_contour(mask)
    contour = np.array([contour[1], contour[0]])
    # ensure start from the point nearest to the bottom extreme point
    dist = np.linalg.norm(np.array(contour.T - np.array(extreme_points[0])), axis=1)
    start_id = np.argmin(dist)
    adjusted_contour = np.concatenate([contour.T[start_id:], contour.T[:start_id]], axis=0)

    gt_vertices = es.distribute_points(adjusted_contour.T, 128)
    gt_vertices = gt_vertices.T
    if isDraw:
        draw_poly_vertices(gt_vertices, img)
        plt.show()
    return gt_vertices

def organize_hdf5_dataset(root, image_root, ann_root, item_list_name):
    isDraw = False
    obj_id = 1
    item_list_path = os.path.join(root, item_list_name + '.txt')
    f = open(item_list_path)
    item_list = f.readlines()
    f.close()
    f_list = open(os.path.join(root, 'img_list_' + item_list_name + '.txt'), 'w')

    for line in tqdm.tqdm(item_list):
        file_name = line.split('.')[0]
        original_img_path = os.path.join(image_root, file_name + '.jpg')
        img = cv2.imread(original_img_path)
        if img is None:
            continue
        seg = read_annotation(os.path.join(ann_root, file_name + '.png'))
        if (seg != 1).all():
            continue
        bboxes = get_bbox(seg, img, isDraw)
        extreme_points = find_extreme_point(bboxes[0], seg, obj_id, img, isDraw)
        octagon_points = get_octagon_points(extreme_points, bboxes[0], img, isDraw)
        init_vert = get_init_vertices(extreme_points, octagon_points, img, isDraw)
        if(np.isnan(init_vert).any()):
            print('nan found')
            pass
        gt_vert = get_gt_vertices(extreme_points, img, seg, isDraw)
        width = img.shape[1]
        height = img.shape[0]
        img_size = [width, height]
        img = transform(img).numpy()
        f_list.write(original_img_path + '\n')

        img = np.array([img])
        gt_vert = np.array([gt_vert])
        init_vert = np.array([init_vert])
        img_size = np.array([img_size])
        save_file_path = os.path.join(root, item_list_name + '.hdf5')
        if not os.path.exists(save_file_path):
            with h5py.File(save_file_path, 'w') as f:
                f.create_dataset("img", data=img, compression="gzip", chunks=True, maxshape=(None, 3, 224, 224))
                f.create_dataset("gt_vert", data=gt_vert, compression="gzip", chunks=True, maxshape=(None, 128, 2))
                f.create_dataset("init_vert", data=init_vert, compression="gzip", chunks=True,
                                 maxshape=(None, 128, 2))
                f.create_dataset("img_size", data=img_size, compression="gzip", chunks=True, maxshape=(None, 2))
        else:
            with h5py.File(save_file_path, 'a') as f:
                f["img"].resize((f["img"].shape[0] + img.shape[0]), axis=0)
                f["img"][-img.shape[0]:] = img

                f["gt_vert"].resize((f["gt_vert"].shape[0] + gt_vert.shape[0]), axis=0)
                f["gt_vert"][-gt_vert.shape[0]:] = gt_vert

                f["init_vert"].resize((f["init_vert"].shape[0] + init_vert.shape[0]), axis=0)
                f["init_vert"][-init_vert.shape[0]:] = init_vert

                f["img_size"].resize((f["img_size"].shape[0] + img_size.shape[0]), axis=0)
                f["img_size"][-img_size.shape[0]:] = img_size
    f_list.close()

def get_model_arguments():
    parser = ArgumentParser()

    # parametrize the network
    parser.add_argument('-rt', '--dataset_root', type=str, default=r'E:\DTU\Jan_2022\Dataset\Oxford')
    parser.add_argument('--organize', action='store_true')

    return parser

if __name__ == '__main__':
    parser = get_model_arguments()
    args = parser.parse_args()
    isDraw = not args.organize

    root = r'E:\DTU\Jan_2022\Dataset\Oxford'
    img_root = os.path.join(root, 'images')
    ann_root = os.path.join(root, 'annotations', 'trimaps')
    if not isDraw:
        organize_hdf5_dataset(root, img_root, ann_root, 'train')
        organize_hdf5_dataset(root, img_root, ann_root, 'val')

    else:
        file_name = 'Abyssinian_2'
        obj_id = 1
        file_path = os.path.join(root, 'images', file_name + '.jpg')
        img = cv2.imread(file_path)
        ann_path = os.path.join(root, 'annotations', 'trimaps', file_name + '.png')
        seg = read_annotation(ann_path)

        bboxes = get_bbox(seg, img, isDraw)
        extreme_points = find_extreme_point(bboxes[0],seg,obj_id,img, isDraw)
        octagon_points = get_octagon_points(extreme_points, bboxes[0], img, isDraw)
        init_vert = get_init_vertices(extreme_points, octagon_points, img, isDraw)
        gt_vert = get_gt_vertices(extreme_points, img, seg, isDraw)
        if isDraw:
            draw_poly_segments(octagon_points, img)
            draw_poly_segments(gt_vert[:int(len(gt_vert)/2)], img)
            plt.imshow(seg)
            plt.axis('off')
            plt.show()
        pass
