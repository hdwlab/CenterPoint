#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Human Dataware Lab. Co., Ltd.
# Created by Daiki Hayashi (hayashi.daiki@hdwlab.co.jp)

import math
import numpy as np
import ros_numpy

from meti_pegasus_msgs.msg import LidarDetectedObject
from meti_pegasus_msgs.msg import LidarDetectedObjectArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Vector3
from jsk_recognition_msgs.msg import BoundingBox
from jsk_recognition_msgs.msg import BoundingBoxArray
from sensor_msgs.msg import PointCloud2


##############################
# BBOX ARRAY MESSAGE RELATED #
##############################
def make_message_bounding_box_array(bbs, header):
    """Make bounding box message from list of bounding boxes.

    Arg:
        bbs (list): list of estimated 3d bounding box with the format
            [class, x, y, z, h, w, l, r, score],
            where (x, y, z) represent the center position of bounding box in lidar
            coordinate, (h, w, l) represent the lenght of sides of bounding box,
            and r represent rotation degree in radian. the shape is (#bbox, 9).
    Return:
        (BoundingBoxArray): ROS message
    """
    msg_out = BoundingBoxArray()
    msg_out.header = header

    for bb in bbs:
        msg_out.boxes.append(make_message_bounding_box(bb, header))

    return msg_out


def make_message_bounding_box(bb, header):
    """Make bounding box message from bounding box.

    Arg:
        bb (list): estimated 3d bounding box with the format
            [class, x, y, z, h, w, l, r, score],
            where (x, y, z) represent the center position of bounding box in lidar
            coordinate, (h, w, l) represent the lenght of sides of bounding box,
            and r represent rotation degree in radian. the shape is (#bbox, 9).
    Return:
        (BoundingBox): ROS message
    """
    c, x, y, z, h, w, l, r, score = bb
    bounding_box = BoundingBox()
    bounding_box.header = header
    bounding_box.pose = make_message_pose(bb)
    bounding_box.dimensions.x = l
    bounding_box.dimensions.y = w
    bounding_box.dimensions.z = h

    return bounding_box


def make_message_pose(bb):
    """Make bounding box message from bounding box.

    Arg:
        bb (list): estimated 3d bounding box with the format
            [class, x, y, z, h, w, l, r, score],
            where (x, y, z) represent the center position of bounding box in lidar
            coordinate, (h, w, l) represent the lenght of sides of bounding box,
            and r represent rotation degree in radian. the shape is (#bbox, 9).
    Return:
        (BoundingBox): ROS message
    """
    c, x, y, z, h, w, l, r, score = bb
    quaternion = to_quaternion(0, 0, r)
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.x = quaternion["x"]
    pose.orientation.y = quaternion["y"]
    pose.orientation.z = quaternion["z"]
    pose.orientation.w = quaternion["w"]

    return pose


def to_quaternion(roll, pitch, yaw):
    """Convert rpy to quaternion.

     NOTE:
         This function assumes that roll, pitch, yaw is defined in ROS coordinate

     Args:
         roll (float): roll
         pitch (float): pitch
         yaw (float): yaw

     Return:
         (dict): dict for quaternion containing w, x, y, z

     """
    q = {"x": None, "y": None, "z": None, "w": None}

    cy = np.cos(-yaw * 0.5)
    sy = np.sin(-yaw * 0.5)
    cr = np.cos(-roll * 0.5)
    sr = np.sin(-roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    q["w"] = cy * cr * cp + sy * sr * sp
    q["x"] = cy * sr * cp - sy * cr * sp
    q["y"] = cy * cr * sp + sy * sr * cp
    q["z"] = sy * cr * cp - cy * sr * sp

    return q


def to_rpy(w, x, y, z):
    """Convert quaternion to rpy.

    NOTE:
        This function assumes that quaternion is defined in ROS coordinate
    
    Args:
        quaternion parameters (w, x, y, z)
        
    Returns:
        roll (float): roll
        pitch (float): pitch
        yaw (float): yaw

    """

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return -X, Y, -Z


#########################################
# LIDAR DETECTED OBJECT MESSAGE RELATED #
#########################################
def make_message_lidar_detected_object_array(bbs, header, version=''):
    """Make lidar detected object message.

    Args:
        bbs (list): list of estimated 3d bounding box with the format
            [class, x, y, z, h, w, l, r, score],
            where (x, y, z) represent the center position of bounding box in lidar
            coordinate, (h, w, l) represent the lenght of sides of bounding box,
            and r represent rotation degree in radian. the shape is (#bbox, 9).
        header (std_msgs.msg.Header): header of the message
        version (str): version
    Return:
        (LidarDetectedObjectArray): ROS message
    """
    msg = LidarDetectedObjectArray()
    msg.header = header
    msg.version = str(version)

    for idx, bb in enumerate(bbs):
        msg.objects.append(make_message_lidar_detected_object(
            bb, header, id=idx, version=version))

    return msg


def make_message_lidar_detected_object(bb, header, id=0, version=''):
    """Make lidar detected object message

    Args:
        bb (list): estimated 3d bounding box with the format
            [class, x, y, z, h, w, l, r, score],
            where (x, y, z) represent the center position of bounding box in lidar
            coordinate, (h, w, l) represent the lenght of sides of bounding box,
            and r represent rotation degree in radian. the shape is (#bbox, 9).
        header (std_msgs.msg.Header): header of the message
        id (int): id of the cluster
        version (str): version
    Return:
        (LidarDetectedObject): ROS message
    """
    # extract bb
    c, x, y, z, h, w, l, r, score = bb

    # init message
    msg = LidarDetectedObject()
    msg.header = header

    # pose
    msg.pose = make_message_pose(bb)

    # cloud
    # points_extracted = self.extract_points_in_box(points, bb)
    # cloud = self.ndarray_to_pointcloud2(points_extracted)
    # msg.cloud = cloud

    # size
    size = Vector3()
    size.x, size.y, size.z = l, w, h
    msg.size = size

    # variance
    variance = Vector3()
    msg.variance = variance

    # feature
    # TODO(d-hayashi): add feature

    # id
    msg.id = id

    # label
    msg.label = c

    # version
    msg.version = str(version)

    # score
    msg.score = score

    return msg


def ndarray_to_pointcloud2(points, header):
    """Make pointcloud2 message from numpy array.

    Arg:
        points (ndarray): point cloud data with shape (#points, 4)
        header (std_msgs.msg.Header): header of the message
    Return:
        (PointCloud2): message
    """
    frame_id = header.frame_id
    stamp = header.stamp

    data = np.zeros(len(points), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('i', np.uint8)
    ])
    data['x'] = points[:, 0]
    data['y'] = points[:, 1]
    data['z'] = points[:, 2]
    data['i'] = points[:, 3]

    msg = ros_numpy.msgify(PointCloud2, data, stamp=stamp, frame_id=frame_id)

    return msg


def remove_points_in_boxes(points, bbs, score_threshold=0.5, margin=0.5):
    """Remove points which are inside the given bbox.
    Arg:
        points (ndarray): input point cloud data with the shape (#points, 4),
            where each dimension represents x, y, z and intensity.
        bbs (list): 3d bounding box with the format [[class, x, y, z, h, w, l, r, score]]
        score_threshold (float): threshold of the score to use the bbox
        margin (float): margin to remove points [m]
    Return:
        (ndarray): input point cloud data with the shape (#points, 4),
            where each dimension represents x, y, z and intensity.
    """
    indices = np.ones(shape=(points.shape[0],), dtype=np.bool)

    for bb in bbs:
        # extract bb
        c, x, y, z, h, w, l, r, score = bb

        # threshold
        if score < score_threshold:
            continue

        # rotate points
        # theta = (-1.0) * r
        theta = r

        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                                    [np.sin(theta), np.cos(theta), 0, 0],
                                    [0, 0, 1.0, 0],
                                    [0, 0, 0, 1.0]])
        points_transformed = np.dot(rotation_matrix, points.T).T - np.array([x, y, z, 0])
        indice = np.all([points_transformed[:, 0] > (-l / 2.0) - margin,
                         points_transformed[:, 0] < (l / 2.0) + margin,
                         points_transformed[:, 1] > (-w / 2.0) - margin,
                         points_transformed[:, 1] < (w / 2.0) + margin,
                         points_transformed[:, 2] > (-h / 2.0) - margin,
                         points_transformed[:, 2] < (h / 2.0) + margin], axis=0)
        indices = np.logical_and(indices, np.logical_not(indice))

    points_processed = points[indices]

    return points_processed
