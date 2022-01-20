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
import tqdm, os
import matplotlib.pyplot as plt

def train(args, net):
    net.cuda()
    criterion.cuda()
    best_val_loss = 100.0
    val_time = 0
    for epoch in range(args.num_epoch):
        running_loss = 0.0
        for i, data in enumerate(tqdm.tqdm(train_loader), 0):
            net.train()
            # get the inputs
            inputs = data['img']
            gt_vertices = data['gt_vert']
            init_vertices = data['init_vert']
            img_sizes = data['img_size']

            # wrap them in Variable
            inputs, gt_vertices, init_vertices, img_sizes = Variable(inputs), Variable(gt_vertices), Variable(init_vertices), Variable(img_sizes)

            # forward + backward + optimize
            output = net(inputs.cuda(), init_vertices.cuda(), img_sizes.cuda())
            loss = criterion(output[-1], gt_vertices.cuda())
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), 20)
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 1000 mini-batches
                writer.add_scalar('training loss', running_loss / 100,
                                  epoch*len(train_loader)+i)
                # writer.close()
                PATH = './runs/' + args.run_name + '/checkpoint/latest.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': running_loss,
                }, PATH)
                running_loss = 0.0

        # Validation
        net.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_time += 1
            for val_data in validation_loader:
                val_inputs = val_data['img']
                val_gt_vert = val_data['gt_vert']
                val_init_vert = val_data['init_vert']
                val_img_size = val_data['img_size']
                val_inputs, val_gt_vert, val_init_vert, val_img_size = Variable(val_inputs), Variable(val_gt_vert), Variable(val_init_vert), Variable(val_img_size)
                val_output = net(val_inputs.cuda(), val_init_vert.cuda(), val_img_size.cuda())
                val_loss += criterion(val_output[-1], val_gt_vert.cuda()).item()
            val_loss /= len(validation_loader)
            writer.add_scalar('validation loss', val_loss,
                              val_time)
            # writer.close()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                PATH = './runs/' + args.run_name + '/checkpoint/best.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': running_loss,
                }, PATH)


    # Save Validation id and draw some result
    # net.eval()
    # with torch.no_grad():
    #     correctClassNum = 0
    #     classNum = 0
    #     totalNum = 0
    #     correctNum = 0
    #     val_loss = 0.0
    #     val_time += 1
    #     path_file = open(r'E:\DL_data\HELMET\validation.txt','w')
    #     for val_data in validation_loader:
    #         val_inputs, val_labels, path = val_data
    #         val_inputs, val_labels = Variable(val_inputs), Variable(val_labels)
    #         val_output = net(val_inputs.cuda())
    #         val_loss += criterion(val_output, val_labels.cuda()).item()
    #         for batch_id, single_val_out in enumerate(val_output):
    #             # path_file.write(path[batch_id] + '\n')
    #             classNum+=1
    #             allCorrectFlag = True
    #             for id, output in enumerate(single_val_out):
    #                 totalNum += 1
    #                 if(abs(output.item()-val_labels[batch_id][id].item())<0.5):
    #                     correctNum+=1
    #                 else:
    #                     allCorrectFlag = False
    #             if allCorrectFlag:
    #                 correctClassNum+=1
    #     val_loss /= len(validation_loader)
    #     print('Final validation loss: ', val_loss, '\nPrecision: ', correctNum / totalNum, '\nClassPrecision: ', correctClassNum/classNum)
    #     path_file.close()
    print('Finished Training')

def get_model_arguments():
    parser = ArgumentParser()

    # parametrize the network
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-mf', '--move_factor', type=int, default=4)
    parser.add_argument('-ne', '--num_epoch', type=int, default=150)
    parser.add_argument('-rs', '--random_seed', type=int, default=8)
    parser.add_argument('-sr', '--split_ratio', type=float, default=0.2)
    parser.add_argument('-dr', '--dataset_root', type=str, default=r'E:\DTU\Jan_2022\Dataset\Oxford')
    parser.add_argument('-rn', '--run_name', type=str, default=r'Oxford')
    parser.add_argument('-lc', '--load_checkpoint', type=str, default=None)
    parser.add_argument('-ns', '--num_sample', type=int, default=40)


    parser.add_argument('--shuffle', action='store_true')

    return parser

if __name__ == '__main__':
    parser = get_model_arguments()
    args = parser.parse_args()

    np.random.seed(args.random_seed)

    # Parameters
    input_channels = 3

    # Tensorboard Setting
    save_path = './runs/' + args.run_name
    if not os.path.exists(save_path + '/checkpoint'):
        os.makedirs(save_path + '/checkpoint')
    writer = SummaryWriter(save_path)

    # Data Prepare
    train_dataset = DataLoader.hdf5Dataset(args.dataset_root, True)
    valid_dataset = DataLoader.hdf5Dataset(args.dataset_root, False)
    train_dataset_size = len(train_dataset)
    valid_dataset_size = len(valid_dataset)
    # indices = list(range(dataset_size))
    # split = int(np.floor(args.split_ratio * dataset_size))
    # if args.shuffle:
    #     np.random.seed(args.random_seed)
    #     np.random.shuffle(indices)
    # train_indices, val_indices = indices[split:], indices[:split]
    #
    # train_sampler = SubsetRandomSampler(train_indices)
    # valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

    net = res_halfSnake.res_halfSnake(args.move_factor)
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint)
        net.load_state_dict(checkpoint['model_state_dict'])


    learning_rate = args.learning_rate
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    train(args, net)