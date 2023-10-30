#!/usr/local/bin/python
# -*- coding: gbk -*-
#============================================================
# SPIRALDATA.PY                -- by Dr. ZhuoQing 2020-11-23
#
# Note:
#============================================================

import argparse
import math
import numpy as np
from point_cloud.experiments.dataloaders import *
from matplotlib import pyplot as plt
import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_point = rotation_matrix.dot(point)
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
    points = math.radians(start) + np.sqrt(np.random.rand(samples, 1)) * math.radians(end)

    # Apply a rotation to the points.
    rotated_x_axis = np.cos(points) * points + np.random.rand(samples, 1) * noise
    rotated_y_axis = np.sin(points) * points + np.random.rand(samples, 1) * noise

    # Stack the vectors inside a samples x 2 matrix.
    rotated_points = np.column_stack((rotated_x_axis, rotated_y_axis))
    return np.apply_along_axis(rotate_point, 1, rotated_points, math.radians(angle))

def main():
    parser = argparse.ArgumentParser(description='Generate n-arm spiral')
    parser.add_argument('--count', type=int, help='Number of samples to generate per arm', default=120)
    parser.add_argument('--arms', type=int, help='Number of args to generate', default=2)
    parser.add_argument('--angle', type=float, help='Angle between each arm.', default=180)
    parser.add_argument('--auto-angle', type=bool, default=True,
                        help='Automatically choose the angle for the arms')
    parser.add_argument('--start', type=float, help='Start angle of the arms', default=60)
    parser.add_argument('--end', type=float, default=300,
                        help='End angle of the arms. A value of 360 corresponds \
                              to a full circle.')
    parser.add_argument('--noise', type=float, help='Noise for the arms', default=0.6)
    parser.add_argument('--filename', type=str, help='Name of the file to save the dataset',
                        default='n_arm_spiral')

    args = parser.parse_args()

    # Create a list of the angles at which to rotate the arms.
    # Either we find the angles automatically by dividing by the number of arms
    # Or we just use the angle given by the user.
    classes = np.empty((0, 3))
    angles = [((360 / args.arms) if args.auto_angle else args.angle) * i for i in range(args.arms)]

    for i, angle in enumerate(angles):
        points = generate_spiral(args.count, args.start, args.end, angle, args.noise)
        classified_points = np.hstack((points, np.full((args.count, 1), i)))
        classes = np.concatenate((classes, classified_points))

        if i == 0:      cstr = 'r'
        elif i == 1:    cstr = 'b'
        elif i ==2:     cstr = 'g'
        else:           cstr = 'yellow'

        plt.scatter(points[: , 0], points[:, 1], color=cstr, alpha=0.6, linewidths=0, s=80)

    # 一个外部圆的操作
    inner_circle = 3*math.pi
    outer_circle = inner_circle+0.5*math.pi

    circle_data = []
    for _ in range(int(args.count*1.4)):
        circle_data.append(np.array(random_point_in_sphere(2, inner_circle, outer_circle).data))

    circle_data = np.array(circle_data)
    plt.scatter(circle_data[:, 0], circle_data[:, 1], color='orange', alpha=0.6, linewidths=0, s=80)

    print(classes.shape)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig("spiral_result.jpg", transparent=True, bbox_inches='tight', pad_inches=0, dpi=1200)
    plt.show()

#    np.savetxt(args.filename + '.csv', classes, fmt=['%10.15f', '%10.15f', '%i'], delimiter=';')

if __name__ == '__main__':
    main()

#------------------------------------------------------------
#        END OF FILE : SPIRALDATA.PY
#============================================================
