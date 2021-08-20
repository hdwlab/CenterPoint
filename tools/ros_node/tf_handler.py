#!/usr/bin/env python

#
# TF handler
#

import rospy
import tf2_ros
from tf2_msgs.msg import TFMessage
from threading import Lock


class TFHandler(object):
    def __init__(self, queue_size=100):
        super(TFHandler, self).__init__()
        self.queue_size = queue_size
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.subscriber = rospy.Subscriber("/tf", TFMessage, self.callback_tf, queue_size=queue_size)
        self.transformations = dict()
        self.frame_ids = []
        self.count = 0
        self.lock = Lock()

    def cache_tf(self):
        frame_pairs = {frame_1.replace("/", ""): [frame_2.replace("/", "") for frame_2 in self.frame_ids if frame_1 != frame_2] for frame_1 in self.frame_ids}
        for parent_frame in frame_pairs.keys():
            if parent_frame not in self.transformations.keys():
                self.transformations.update({parent_frame: dict()})
            for child_frame in frame_pairs[parent_frame]:
                if child_frame in self.transformations[parent_frame].keys():
                    continue
                self.transformations[parent_frame][child_frame] = \
                    self.tf_buffer.lookup_transform(parent_frame, child_frame, rospy.Time())

    def get_tf(self, parent_frame, child_frame):
        if parent_frame in self.transformations.keys():
            if child_frame in self.transformations[parent_frame].keys():
                return self.transformations[parent_frame][child_frame]

        return self.tf_buffer.lookup_transform(parent_frame, child_frame, rospy.Time())

    def callback_tf(self, msg):
        self.count += 1
        if self.count > self.queue_size:
            self.subscriber.unregister()
        for tf in msg.transforms:
            if tf.header.frame_id not in self.frame_ids:
                self.frame_ids.append(tf.header.frame_id)
            if tf.child_frame_id not in self.frame_ids:
                self.frame_ids.append(tf.child_frame_id)
        with self.lock:
            self.cache_tf()


if __name__ == "__main__":
    rospy.init_node("test")
    test = TFHandler()
    rospy.spin()
