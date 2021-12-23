#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ROS node of second-based 3d object detection model
#
# Copyright 2018 Human Dataware Lab. Co., Ltd.
# Created by Tomoki Hayashi (hayashi@hdwlab.co.jp)

import argparse
import os
import sys
import threading
import time

from queue import Queue

import numpy as np
import ros_numpy
import rospy
import torch

from google.protobuf import text_format

from ros_node.node_base import BaseObjectDetectorHandler
from second.protos import pipeline_pb2
from second.pytorch.train import build_network
from second.utils import config_tool

from ros_node.utils import make_message_bounding_box_array, \
    make_message_lidar_detected_object_array, remove_points_in_boxes


class SecondObjectDetectorHandler(BaseObjectDetectorHandler):
    """Object detector handlar using Second models.

    Args:
        model_ckpt (str): model checkpoint path.
        model_conf (str): model config path (default=None).
        max_queue_size (int): maximum queue size (default=64).
        remove_points_in_objects (bool): set False to avoid publishing points_without_objects
        
    """

    def __init__(self, model_ckpt, model_conf=None, max_queue_size=64, 
                 remove_points_in_objects=True, **kwargs):
        super(SecondObjectDetectorHandler, self).__init__(**kwargs)
        
        # substitute
        self.points_removal = remove_points_in_objects
        
        # store hyper parameters
        self.model_ckpt = model_ckpt
        if model_conf is not None:
            self.model_conf = model_conf
        else:
            self.model_conf = os.path.dirname(model_ckpt) + "/pipeline.config"
        self.max_queue_size = max_queue_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.version = 0.2

        # load config file
        rospy.logdebug("load model config from %s." % self.model_conf)
        cfg = self._load_model_conf(self.model_conf)
        model_cfg = cfg.model.second

        # build network
        rospy.logdebug("load model parameters from %s." % self.model_ckpt)
        self.net = build_network(model_cfg).to(self.device).eval()
        self.net.load_state_dict(torch.load(self.model_ckpt))
        self.target_assigner = self.net.target_assigner
        self.voxel_generator = self.net.voxel_generator
        self.class_names = self.target_assigner.classes

        # generate anchors
        grid_size = self.voxel_generator.grid_size
        feature_map_size = grid_size[:2] // config_tool.get_downsample_factor(model_cfg)
        feature_map_size = [*feature_map_size, 1][::-1]
        rospy.logdebug("grid size: %s" % str(grid_size))
        rospy.logdebug("feature map size: %s" % str(feature_map_size))
        anchors = self.target_assigner.generate_anchors(feature_map_size)["anchors"]
        anchors = torch.tensor(anchors, dtype=torch.float32, device=self.device)
        self.anchors = anchors.view(1, -1, 7)

        # setup queue
        self.setup_queues()
        self.is_stop = False
        self.num_queues = 0

        # start each daemon
        self.threads = []
        self.threads += [threading.Thread(target=self.run_preprocess)]
        self.threads += [threading.Thread(target=self.run_inference)]
        self.threads += [threading.Thread(target=self.run_postprocess)]
        self.threads += [threading.Thread(target=self.run_publish)]
        for thread in self.threads:
            thread.setDaemon(True)
            thread.start()

    def callback(self, msg):
        if self.num_queues >= self.max_queue_size:
            return
        self.counter += 1
        self.msg_queue.put(msg)
        self.num_queues += 1

    @staticmethod
    def _load_model_conf(model_conf):
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(model_conf, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)

        return config

    def setup_queues(self):
        self.msg_queue = Queue(maxsize=self.max_queue_size)
        self.example_queue = Queue(maxsize=self.max_queue_size)
        self.bbox_queue = Queue(maxsize=self.max_queue_size)
        self.output_queue = Queue(maxsize=self.max_queue_size)

    def run_preprocess(self):
        while not self.is_stop:
            start_time = time.time()
            msg = self.msg_queue.get()
            raw_points = self._convert_msg_to_points(msg) if self.points_removal else None
            header = msg.header
            example = self.preprocess(msg)
            self.example_queue.put((example, header, raw_points))
            rospy.logdebug("preprocess required time = %s sec." % (time.time() - start_time))

    def run_inference(self):
        while not self.is_stop:
            start_time = time.time()
            example, header, raw_points = self.example_queue.get()
            bboxes, scores, labels, raw_points = self.inference(example, raw_points)
            self.bbox_queue.put(((bboxes, scores, labels), header, raw_points))
            rospy.logdebug("inference required time = %s sec." % (time.time() - start_time))

    def run_postprocess(self):
        while not self.is_stop:
            start_time = time.time()
            preds, header, raw_points = self.bbox_queue.get()
            points_without_objects = self.postprocess(*preds, raw_points)
            self.output_queue.put((preds, header, points_without_objects))
            rospy.logdebug("postprocess required time = %s sec." % (time.time() - start_time))

    def run_publish(self):
        while not self.is_stop:
            start_time = time.time()
            preds, header, points_without_objects = self.output_queue.get()
            self.publish(*preds, header, points_without_objects)
            self.num_queues -= 1
            rospy.logdebug("publish required time = %s sec." % (time.time() - start_time))

    def preprocess(self, msg):
        """Preprocess to convert ROS message to the input dict"""
        points = self._convert_msg_to_points(msg)
        temp = self.voxel_generator.generate(points, max_voxels=90000)
        voxels, coords, num_points = temp['voxels'], temp['coordinates'], temp['num_points_per_voxel']
        rospy.logdebug("voxel shape: %s" % str(voxels.shape))
        # add batch idx to coords
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)
        example = {
            "anchors": self.anchors,
            "voxels": voxels,
            "num_points": num_points,
            "coordinates": coords,
        }

        return example

    def inference(self, example, raw_points):
        """Inference."""
        pred = self.net(example)[0]
        bboxes = pred["box3d_lidar"].detach().cpu().numpy()
        scores = pred["scores"].detach().cpu().numpy()
        labels = pred["label_preds"].detach().cpu().numpy()
        # TODO(kan-bayashi): check each value in the box3d_lidar

        return bboxes, scores, labels, raw_points

    def postprocess(self, bboxes, scores, labels, raw_points):
        """Post-process."""
        # remove points of objects from raw_points
        bbs = []
        for bbox, score, label_idx in zip(bboxes, scores, labels):
            x, y, z, l, w, h, r = bbox
            bb = [self.class_names[int(label_idx)], x, y, z, h, w, l, r, float(score)]
            bbs.append(bb)
        if self.points_removal:
            points_without_objects = remove_points_in_boxes(
                raw_points, 
                bbs,
                score_threshold=rospy.get_param("~score_threshold", 0.3),
                margin=rospy.get_param("~point_removal_margin", 0.5)
            )
        else:
            points_without_objects = raw_points
        return points_without_objects

    def publish(self, bboxes, scores, labels, header, processed_points):
        """Publish ROS messages"""
        # re-format bbox information
        publish_bboxes = []
        for bbox, score, label_idx in zip(bboxes, scores, labels):
            x, y, z, l, w, h, r = bbox
            publish_bboxes += [[self.class_names[int(label_idx)], x, y, z, h, w, l, r, score]]

        bbox_array_msg = make_message_bounding_box_array(publish_bboxes, header)
        self.publisher_bouding_box_array.publish(bbox_array_msg)

        lidar_obj_msg = make_message_lidar_detected_object_array(publish_bboxes, header)
        self.publisher_lidar_detected_object_array.publish(lidar_obj_msg)

        if self.points_removal:
            points_msg = self._convert_points_to_msg(processed_points, header)
            self.publisher_points_without_objects.publish(points_msg)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", type=str, required=True,
                        help="model checkpoint filepath.")
    parser.add_argument("--model-conf", type=str,
                        help="model config filepath.")
    parser.add_argument("--verbose", action="store_true",
                        help="verbose mode.")
    parser.add_argument("--queue-size", type=int, default=100, 
                        help="Queue size of subscription")
    parser.add_argument("--skip-point-removal", action="store_true",
                        help="Skip publishing points_without_objects")
    return parser.parse_args()


def main():
    args = get_arguments()
    if args.verbose:
        log_level = rospy.DEBUG
    else:
        log_level = rospy.INFO
    rospy.init_node("second", log_level=log_level)
    handler = SecondObjectDetectorHandler(
        model_ckpt=args.model_ckpt,
        model_conf=args.model_conf,
        max_queue_size=args.queue_size,
        remove_points_in_objects=not(args.skip_point_removal),
        queue_size=args.queue_size
    )
    try:
        handler.spin()
    except KeyboardInterrupt:
        handler.is_stop = False
        sys.exit(1)


if __name__ == "__main__":
    main()
