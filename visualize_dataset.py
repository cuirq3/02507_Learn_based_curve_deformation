import h5py, os, cv2
import matplotlib.pyplot as plt
from data_utils.sbd_utils import draw_poly_vertices, get_bbox, process_mat_seg

def visualize_valid(root, file_name, id):
    # bboxes = get_bbox(seg, img)
    h5File = h5py.File(os.path.join(root, file_name), 'r')
    file = open(os.path.join(os.path.join(root, 'img_list_train.txt')))
    img_list = file.readlines()
    file.close()
    gt_vertices = h5File['gt_vert'][id]
    init_vertices = h5File['init_vert'][id]
    img_path = img_list[id].split('\n')[0]
    img = cv2.imread(img_path)
    # seg = process_mat_seg(os.path.join(root, 'cls', img_path.split('\\')[-1].split('.')[0] + '.mat'))
    # get_bbox(seg, img)
    plt.axis('off')
    draw_poly_vertices(gt_vertices, img)
    plt.axis('off')
    draw_poly_vertices(init_vertices, img)

if __name__ == '__main__':
    visualize_valid(r'E:\DTU\Jan_2022\Deep_snake\snake\data\sbd', 'train.hdf5', 2)
    # visualize_valid(r'E:\DTU\Jan_2022\Dataset\Oxford', 'train.hdf5', 2)

