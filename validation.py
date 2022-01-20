from model import res_halfSnake
import torch, torchvision
from data_utils import DataLoader
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import tqdm, os, cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def validate(args, net, out_path):
    net.cuda()
    criterion.cuda()
    val_time = 0
    f = open(os.path.join(args.dataset_root, 'img_list_val.txt'))
    img_list = f.readlines()
    f.close()

    net.eval()
    with torch.no_grad():
        IoU = 0.0
        val_loss = 0.0
        item_num = 0
        for val_data in tqdm.tqdm(validation_loader):
            val_inputs = val_data['img']
            val_gt_vert = val_data['gt_vert']
            val_init_vert = val_data['init_vert']
            val_img_size = val_data['img_size']
            val_inputs, val_gt_vert, val_init_vert, val_img_size = Variable(val_inputs), Variable(
                val_gt_vert), Variable(val_init_vert), Variable(val_img_size)
            val_output = net(val_inputs.cuda(), val_init_vert.cuda(), val_img_size.cuda())
            val_loss += criterion(val_output[-1], val_gt_vert.cuda()).item()
            for batch_num in range(val_output[-1].shape[0]):
                start_points = val_output[-1][batch_num].cpu().numpy().tolist()
                img_path = img_list[val_time]
                img = cv2.imread(img_path.split('\n')[0])
                for id, start_point in enumerate(start_points):
                    plt.plot(start_point[0], start_point[1], marker='o', color='red')
                plt.imshow(img)
                plt.savefig(os.path.join(out_path, str(val_time) + '.jpg'))
                val_time += 1
                plt.clf()

                # calculate IoU
                polygon = list(map(tuple, start_points))
                height, width, _ = img.shape
                img = Image.new('L', (width, height), 0)
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                mask_pred = np.array(img)
                # plt.imshow(mask_pred)
                # plt.show()

                gt_points = val_gt_vert[batch_num].cpu().numpy().tolist()
                polygon = list(map(tuple, gt_points))
                img = Image.new('L', (width, height), 0)
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                mask_gt = np.array(img)
                # plt.imshow(mask_gt)
                # plt.show()

                intersection = np.logical_and(mask_gt, mask_pred)
                union = np.logical_or(mask_gt, mask_pred)
                iou_score = np.sum(intersection) / np.sum(union)
                IoU += iou_score
                item_num += 1

        val_loss /= len(validation_loader)
        IoU /= item_num
        print(val_loss)
        print(IoU)


def get_model_arguments():
    parser = ArgumentParser()

    # parametrize the network
    parser.add_argument('-bs', '--batch_size', type=int, default=16)
    parser.add_argument('-mf', '--move_factor', type=int, default=4)
    parser.add_argument('-dr', '--dataset_root', type=str, default=r'E:\DTU\Jan_2022\Dataset\Oxford')
    parser.add_argument('-lc', '--load_checkpoint', type=str, default=None)
    parser.add_argument('-ns', '--num_sample', type=int, default=40)
    parser.add_argument('-sf', '--save_folder', type=str, default='validation')


    return parser

if __name__ == '__main__':
    parser = get_model_arguments()
    args = parser.parse_args()

    # Parameters
    input_channels = 3

    # Data Prepare
    valid_dataset = DataLoader.hdf5Dataset(args.dataset_root, False)
    valid_dataset_size = len(valid_dataset)
    train_dataset = DataLoader.hdf5Dataset(args.dataset_root, True)
    train_dataset_size = len(train_dataset)

    validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    net = res_halfSnake.res_halfSnake(args.move_factor)
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint)
        net.load_state_dict(checkpoint['model_state_dict'])

    save_path = './runs/' + args.save_folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    criterion = nn.SmoothL1Loss()

    validate(args, net, save_path)