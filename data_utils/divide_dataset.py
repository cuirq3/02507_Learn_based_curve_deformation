import os, random
from argparse import ArgumentParser

def divide_dataset(ratio, img_root, out_root):
    train_file = open(os.path.join(out_root, 'train.txt'), 'w')
    val_file = open(os.path.join(out_root, 'val.txt'), 'w')
    for subdir, dirs, files in os.walk(img_root):
        for file in files:
            if file.endswith('.jpg'):
                if(random.random()>=1-ratio):
                    val_file.write(file + '\n')
                else:
                    train_file.write(file + '\n')

def get_model_arguments():
    parser = ArgumentParser()

    # parametrize the network
    parser.add_argument('-ro', '--split_ratio', type=float, default=0.2)
    parser.add_argument('-rt', '--dataset_root', type=str, default=r'E:\DTU\Jan_2022\Dataset\Oxford')

    return parser

if __name__ == '__main__':
    parser = get_model_arguments()
    args = parser.parse_args()
    divide_dataset(args.split_ratio, os.path.join(args.dataset_root, 'images'), r'E:\DTU\Jan_2022\Dataset\Oxford')