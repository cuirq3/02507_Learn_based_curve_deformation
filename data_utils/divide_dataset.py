import os, random

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

if __name__ == '__main__':
    divide_dataset(0.2, r'E:\DTU\Jan_2022\Dataset\Oxford\images', r'E:\DTU\Jan_2022\Dataset\Oxford')