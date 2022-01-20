import torch
import torch.nn as nn
from . import res_customize
from . import snake
class Deformation(nn.Module):
    def __init__(self, move_factor):
        super(Deformation, self).__init__()
        # self.fuse_res = [nn.Linear(64, 128).cuda(), nn.Linear(128, 128).cuda(), nn.Linear(256, 128).cuda(), nn.Linear(512, 128).cuda(), nn.Linear(4*128, 128).cuda()]
        self.fuse_res = nn.Conv1d(64+128+256+512, 64, 1)
        self.iter = 2
        # self.move_factor = [move_factor / 2] * int(self.iter / 2) + [move_factor] * int(self.iter / 2)
        self.move_factor = 1
        self.init_deform = snake.Snake(state_dim=128, feature_dim=64 + 2, conv_type='dgrid')
        for i in range(self.iter):
            # deform_iter = snake.Snake(state_dim=128, feature_dim=128 + 2, conv_type='dgrid')
            deform_iter = snake.Snake(state_dim=128, feature_dim=64 + 2, conv_type='dgrid')
            self.__setattr__('deform_iter' + str(i), deform_iter)

    def get_relative_coord(self, vertices):
        x_min = torch.min(vertices[:,:,0], dim=1, keepdim=True)[0]
        y_min = torch.min(vertices[:,:,1], dim=1, keepdim=True)[0]
        relative_coord = vertices.clone()
        relative_coord[:,:,0] -= x_min
        relative_coord[:,:,1] -= y_min
        return relative_coord

    def init_vertices_feature(self, res_output, vertices, rel_vertices, width, height):
        init_features = []
        vertices = self.convert_vertices_coord(vertices, width, height)
        sample_coord = vertices * 2 - 1
        for id, single_res in enumerate(res_output):
            point_feature = nn.functional.grid_sample(single_res, sample_coord.unsqueeze(1).float()).squeeze().permute(0,2,1)
            init_features.append(point_feature)
        init_features = self.fuse_res(torch.cat(init_features, dim=2).permute(0,2,1)).permute(0,2,1)
        loc_feature = rel_vertices/self.move_factor/4
        init_features = torch.cat([init_features, loc_feature], dim=2)
        return init_features

    def convert_vertices_coord(self, vertices, width, height):
        processed_vertices = vertices.clone()
        processed_vertices[:,:,0]/=width.unsqueeze(1)
        processed_vertices[:,:,1]/=height.unsqueeze(1)
        return processed_vertices

    def forward(self, res_output, vertices, width, height):
        predicted_vertices = vertices.clone()
        predicted_rel_vertices = self.get_relative_coord(predicted_vertices)
        predictions = []
        vertices_feature = self.init_vertices_feature(res_output, predicted_vertices, predicted_rel_vertices, width,
                                                      height).permute(0, 2, 1).float()
        predicted_offset = self.init_deform(vertices_feature, None).permute(0, 2, 1)
        prediction = predicted_vertices * self.move_factor
        prediction += predicted_offset
        predicted_vertices = prediction / self.move_factor
        predictions.append(predicted_vertices)
        for i in range(self.iter):
            predicted_rel_vertices = self.get_relative_coord(predicted_vertices)
            vertices_feature = self.init_vertices_feature(res_output, predicted_vertices, predicted_rel_vertices, width,
                                                          height).permute(0, 2, 1).float()
            iter_model = self.__getattr__('deform_iter' + str(i))
            predicted_offset = iter_model(vertices_feature, None).permute(0, 2, 1)
            prediction = predicted_vertices * self.move_factor
            prediction += predicted_offset
            predicted_vertices = prediction / self.move_factor
            predictions.append(predicted_vertices)

        return predictions





class res_halfSnake(nn.Module):

    def __init__(self, move_factor):
        super(res_halfSnake, self).__init__()

        self.res = res_customize.resnet18(True)
        self.halfSnake = Deformation(move_factor)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, img, vertices, img_size):
        width = img_size[:,0]
        height = img_size[:,1]
        _, res_output = self.res(img)
        predicted_vertices = self.halfSnake(res_output, vertices, width, height)
        return predicted_vertices

if __name__ == '__main__':
    model = res_halfSnake()
    print(model)


