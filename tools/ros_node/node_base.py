#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Human Dataware Lab. Co., Ltd.
# Created by Tomoki Hayashi (hayashi@hdwlab.co.jp)

import math
import numpy as np
import ros_numpy
import rospy

from meti_pegasus_msgs.msg import LidarDetectedObjectArray
from jsk_recognition_msgs.msg import BoundingBoxArray
from sensor_msgs.msg import PointCloud2


class BaseObjectDetectorHandler(object):
    """Base object detector handler class for ROS"""

    def __init__(self,
                 queue_size=100):
        """Initialiation of subscriber and publisher.

        Args:
            bbox_topic_name (str): name of bounding box array topic.
            queue_size (int): default queue size for ros node.
        """
        self.subscriber = rospy.Subscriber(
            rospy.get_param("~topic_in", "/points_raw"), PointCloud2,
            self.callback, queue_size=queue_size
        )
        self.publisher_bouding_box_array = rospy.Publisher(
            "/bounding_box", BoundingBoxArray, queue_size=queue_size
        )
        self.publisher_lidar_detected_object_array = rospy.Publisher(
            "/detected_objects", LidarDetectedObjectArray, queue_size=queue_size
        )
        self.publisher_points_without_objects = rospy.Publisher(
            "/points_without_objects", PointCloud2, queue_size=queue_size
        )
        self.counter = 0

    def spin(self):
        """Spin this node."""
        rospy.spin()

    def publish(self, *args, **kwargs):
        """Publish detected object messages."""
        raise NotImplementedError()

    def callback(self, msg):
        """Callback when get topic messages"""
        raise NotImplementedError

    @staticmethod
    def _convert_msg_to_points(msg):
        points = ros_numpy.numpify(msg)[["x", "y", "z", "intensity"]]
        pointcloud = np.array(points.tolist())
        pointcloud[:, 3] /= 255.0           # scale to [0, 1]

        return pointcloud

    @staticmethod
    def _convert_points_to_msg(points, header=None):
        points[:, 3] *= 255.0  # scale to [0, 255]
        new_points = np.core.records.fromarrays(points.transpose(), names='x, y, z, intensity', formats='f, f, f, f')
        msg = ros_numpy.msgify(PointCloud2, new_points)
        if header is not None:
            msg.header = header

        return msg
