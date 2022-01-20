from model import res_halfSnake
import torch
from torchvision import transforms
import numpy as np
from torch.autograd import Variable
import os
from cv2 import cv2
from tqdm import tqdm

transform = transforms.Compose([
    transforms.ToPILImage(),
    # resize
    transforms.Resize((224, 224)),
    # to-tensor
    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def data_read(root, filename):
    path = os.path.join(root, filename)
    f = open(path)
    lines = f.readlines()
    labels = []
    for line in lines:
        item = line.split()
        labels.append(item)

    image = cv2.imread(os.path.join(root, filename.split('.')[0] + '.jpg'))
    f.close()
    return image, labels

def data_read_dir(label_root):
    labels = []
    images = []
    names = []
    for subdir, dirs, files in os.walk(label_root):
        for file in files:
            if file.endswith('.txt'):
                f = open(os.path.join(label_root, file), 'r')
                lines = f.readlines()
                labels_in1 = []
                for line in lines:
                    item = line.split()
                    labels_in1.append(item)
                images.append(os.path.join(r'E:\DL_data\HELMET\images\val', file.split('.')[0] + '.jpg'))
                names.append(file.split('.')[0])
                labels.append(labels_in1)
    return  images, labels, names

def data_process(image, label):
    box = label[1:]
    image = image[int(box[1]):int(box[1]) + int(box[3]), int(box[0]):int(box[0]) + int(box[2])]
    image = transform(image)
    image = torch.Tensor(image)
    return image

def data_process_from_yolo(image, label):
    box = label[1:]
    height, width, _ = image.shape
    x = int(width*float(box[0]))
    y = int(height*float(box[1]))
    w = int(width*float(box[2]))
    h = int(height*float(box[3]))
    image = image[max(y-int(h/2),0):min(y+int(h/2),height), max(x-int(w/2),0):min(x+int(w/2),width)]
    image = transform(image)
    image = torch.Tensor(image)
    return image

def result_translate(output):
    Person_str = ['P0','P1','P2','P3']
    str_result = ''
    digit_result = ''
    output = output[0]
    for digit in output:
        if digit > 0.5:
            digit_result += '1'
        else:
            digit_result += '0'
    if output[4] > 0.5:
        str_result = 'DHelmet'
    else:
        str_result = 'DNoHelmet'
    for id, exist_info in enumerate(output[:4]):
        if exist_info > 0.5:
            str_result += Person_str[id]
            if output[id+5]>0.5:
                str_result+='Helmet'
            else:
                str_result+='NoHelmet'
    return str_result, digit_result

def inference(root, filename, ckpt_path, write_root):
    # Parameters
    input_channels = 3

    net = model.HelmetDetector(input_channels, 64, 64, 7)
    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.cuda()

    # Load data
    image, labels = data_read(root, filename)
    for label in labels:
        input_image = data_process(image, label)
        input_image = torch.unsqueeze(input_image, dim=0)

        # wrap them in Variable
        input_image = Variable(input_image)
        net.eval()
        with torch.no_grad():
            output = net(input_image.cuda())
            str_result = result_translate(output)
            box = label[1:]
            x = int(box[0])
            y = int(box[1])
            w = int(box[2])
            h = int(box[3])
            color = tuple(np.random.randint(0,255,3).tolist())
            image = cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
            image = cv2.putText(image, str_result, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imwrite(os.path.join(write_root, filename.split('.')[0] + '.jpg'), image)

def inference_exhibition(img_path, xyxys, ckpt_path):
    # Parameters
    input_channels = 3

    net = model.HelmetDetector(input_channels, 64, 64, 7)
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['model_state_dict'])

    img = cv2.imread(img_path)
    height, width, _ = img.shape
    img_out = img.copy()

    for xyxy in xyxys:
        image = img[max(int(xyxy[1].item()), 0):min(int(xyxy[3].item()), height), max(int(xyxy[0].item()), 0):min(int(xyxy[2].item()), width)]
        image = transform(image)
        image = torch.Tensor(image)
        input_image = torch.unsqueeze(image, dim=0)

        # wrap them in Variable
        input_image = Variable(input_image)
        net.eval()
        with torch.no_grad():
            output = net(input_image)
            str_result, _ = result_translate(output)
            color = tuple(np.random.randint(0,255,3).tolist())
            img_out = cv2.rectangle(img_out, (int(xyxy[0].item()), int(xyxy[1].item())), (int(xyxy[2].item()), int(xyxy[3].item())), color, 1)
            img_out = cv2.putText(img_out, str_result, (int(xyxy[0].item()), int(xyxy[1].item()) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return img_out


def inference_from_yolo(label_root, ckpt_path, write_root):
    input_channels = 3

    net = model.HelmetDetector(input_channels, 64, 64, 7)
    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.cuda()

    # Load data
    images, labels, names = data_read_dir(label_root)
    for id, image_path in enumerate(tqdm(images)):
        image = cv2.imread(image_path)
        label_per_image = labels[id]
        name = names[id]
        height, width, _ = image.shape
        image_out = image.copy()
        for label in label_per_image:
            input_image = data_process_from_yolo(image, label)
            input_image = torch.unsqueeze(input_image, dim=0)

            # wrap them in Variable
            input_image = Variable(input_image)
            net.eval()
            f = open(os.path.join(write_root, name + '.txt'), 'a')
            with torch.no_grad():
                output = net(input_image.cuda())
                str_result, digit_result = result_translate(output)
                box = label[1:]
                x = int(width * float(box[0]))
                y = int(height * float(box[1]))
                w = int(width * float(box[2]))
                h = int(height * float(box[3]))
                color = tuple(np.random.randint(0, 255, 3).tolist())
                image_out = cv2.rectangle(image_out, (max(x-int(w/2),0), max(y-int(h/2),0)), (min(x+int(w/2),width), min(y+int(h/2),height)), color, 1)
                image_out = cv2.putText(image_out, str_result, (max(x-int(w/2),0), max(y-int(h/2),10)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            f.write(digit_result + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n')
            f.close()
        cv2.imwrite(os.path.join(write_root, name + '.jpg'), image_out)

if __name__ == '__main__':
    # inference(r'E:\DL_data\HELMET\inference_data', '6593.txt', './runs/test/checkpoint/best.pt', r'E:\DL_data\HELMET\inference_output')
    inference_from_yolo(r'E:\DTU\Semester_1\Deep_learning\Final_project\02456_Final_Project\Yolov5\runs\detect\val2\labels','./runs/Final_follow/checkpoint/best.pt',r'E:\DL_data\HELMET\inference_output')
