# Adapted from https://github.com/EmilienDupont/augmented-neural-odes/blob/master/experiments/dataloaders.py
import math

import numpy as np
import torch
from random import random
from torch.utils.data import Dataset


class ConcentricSphere(Dataset):
    """Dataset of concentric d-dimensional spheres. Points in the inner sphere
    are mapped to -1, while points in the outer sphere are mapped 1.
    Parameters
    ----------
    dim : int
        Dimension of spheres.
    inner_range : (float, float)
        Minimum and maximum radius of inner sphere. For example if inner_range
        is (1., 2.) then all points in inner sphere will lie a distance of
        between 1.0 and 2.0 from the origin.
    outer_range : (float, float)
        Minimum and maximum radius of outer sphere.
    num_points_inner : int
        Number of points in inner cluster
    num_points_outer : int
        Number of points in outer cluster
    """

    def __init__(self, dim, inner_range, outer_range, num_points_inner,
                 num_points_outer):
        self.dim = dim
        self.inner_range = inner_range
        self.outer_range = outer_range
        self.num_points_inner = num_points_inner
        self.num_points_outer = num_points_outer

        self.data = []
        self.targets = []

        # Generate data for inner sphere
        for _ in range(self.num_points_inner):
            self.data.append(
                random_point_in_sphere(dim, inner_range[0], inner_range[1])
            )
            self.targets.append(torch.Tensor([-1]))

        # Generate data for outer sphere
        for _ in range(self.num_points_outer):
            self.data.append(
                random_point_in_sphere(dim, outer_range[0], outer_range[1])
            )
            self.targets.append(torch.Tensor([1]))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)


class SpiralCircle(Dataset):
    """Dataset of some spiral arms and the outer circle with noise. Points in different area
    will be classified in its class.tips: all kinds of the generated data are 2d-array.
    Parameters
    ----------
    dim : int
        Dimension of arms(plus circle).
    outer_range : (float, float)
        Minimum and maximum radius of outer sphere.
    num_points_spiral: int
        Number of points in spiral arms area
    num_points_outer : int
        Number of points in outer cluster
    """

    def __init__(self, dim, outer_range, num_points_spiral, num_points_outer):
        # 这里限制住旋转臂的角度的变化以及范围
        self.cnt = dim
        self.outer_range = outer_range
        self.num_points_spiral = num_points_spiral
        self.num_points_outer = num_points_outer

        # 构建对应的数据以及标签列表
        self.data = []
        self.targets = []

        # 开始按照要求生产指定的数据
        # 螺旋臂的数目
        self.spiral_arm_cnt = self.cnt - 1
        # 外圈的数目
        self.circle_cnt = 1

        # 对于螺旋臂设置不同的标签
        for i in range(1, self.cnt):
            points = generate_spiral(self.num_points_spiral, 0, 360, 180, 0.3)
            for point in points:
                self.data.append(point)
                tmp_hot = torch.zeros(self.cnt)
                tmp_hot[i-1] = 1
                self.targets.append(tmp_hot)

        # 单独为外圈的数据设置对应的标签
        for _ in range(self.num_points_outer):
            self.data.append(
                random_point_in_sphere(2, self.outer_range[0], self.outer_range[1])
            )
            tmp_hot = torch.zeros(self.cnt)
            tmp_hot[self.cnt - 1] = 1
            self.targets.append(tmp_hot)

        # 由于是分类问题，所以直接采用one-hot编码结构

        print("spiral-data load success!")

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return len(self.data)


def random_point_in_sphere(dim, min_radius, max_radius):
    """Returns a point sampled uniformly at random from a sphere if min_radius
    is 0. Else samples a point approximately uniformly on annulus.
    Parameters
    ----------
    dim : int
        Dimension of sphere
    min_radius : float
        Minimum distance of sampled point from origin.
    max_radius : float
        Maximum distance of sampled point from origin.
    """
    # Sample distance of point from origin
    unif = random()
    distance = (max_radius - min_radius) * (unif ** (1. / dim)) + min_radius
    # Sample direction of point away from origin
    direction = torch.randn(dim)
    unit_direction = direction / torch.norm(direction, 2)
    return distance * unit_direction


def rotate_point(point, angle):
    """Rotate two point by an angle.
    Parameters
    ----------
    point: 2d numpy array
        The coordinate to rotate.
    angle: float
        The angle of rotation of the point, in degrees.
    Returns
    -------
    2d numpy array
        Rotated point.
    """
    rotation_matrix = torch.Tensor([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_point = torch.matmul(rotation_matrix, torch.Tensor(point))
    return rotated_point


def generate_spiral(samples, start, end, angle, noise):
    """Generate a spiral of points.
    Given a starting end, an end angle and a noise factor, generate a spiral of points along
    an arc.
    Parameters
    ----------
    samples: int
        Number of points to generate.
    start: float
        The starting angle of the spiral in degrees.
    end: float
        The end angle at which to rotate the points, in degrees.
    angle: float
        Angle of rotation in degrees.
    noise: float
        The noisyness of the points inside the spirals. Needs to be less than 1.
    """
    # Generate points from the square root of random data inside an uniform distribution on [0, 1).
    points = torch.Tensor(math.radians(start) + np.sqrt(np.random.rand(samples, 1)) * math.radians(end))

    # Apply a rotation to the points.
    rotated_x_axis = torch.cos(points) * points + torch.rand(samples, 1) * noise
    rotated_y_axis = torch.sin(points) * points + torch.rand(samples, 1) * noise

    # Stack the vectors inside a samples x 2 matrix.
    rotated_points = torch.column_stack((rotated_x_axis, rotated_y_axis))
    return torch.Tensor(np.apply_along_axis(rotate_point, 1, rotated_points, math.radians(angle)))
