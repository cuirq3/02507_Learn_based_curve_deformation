import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os, h5py
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

transform = transforms.Compose([
    transforms.ToPILImage(),
    # resize
    transforms.Resize((224, 224)),
    # to-tensor
    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class hdf5Dataset(Dataset):
    def __init__(self, root, isTrain):
        super(Dataset, self).__init__()
        file_path = os.path.join(root, 'train.hdf5' if isTrain else 'val.hdf5')
        self.data = h5py.File(file_path, 'r')

    def __len__(self):
        return self.data['img'].shape[0]

    def __getitem__(self, index):
        img = torch.tensor(self.data['img'][index])
        gt_vertices = torch.tensor(self.data['gt_vert'][index])
        init_vertices = torch.tensor(self.data['init_vert'][index])
        img_size = torch.tensor(self.data['img_size'][index])
        return {'img': img,
                # 'seg': seg,
                # 'bboxes':bboxes,
                'gt_vert':gt_vertices,
                'init_vert':init_vertices,
                'img_size':img_size}


if __name__ == '__main__':
    dataset = hdf5Dataset(r'E:\DTU\Jan_2022\Deep_snake\snake\data\sbd', True)
    for idx, item in enumerate(dataset):
        data = item
        print(idx)
    pass