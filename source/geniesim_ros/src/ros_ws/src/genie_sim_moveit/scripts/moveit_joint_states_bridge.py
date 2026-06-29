#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Bridge the simulator-side joint state graph into MoveIt.

Two responsibilities, both downstream of ``/joint_states`` and both
required for MoveIt's PlanningSceneMonitor to report a complete robot
state:

  1. **QoS bridge**.  The simulator publishes ``/joint_states`` with
     SensorData QoS (BEST_EFFORT, KEEP_LAST 5).  MoveIt subscribes
     with Reliable QoS by default, and the QoS handshake silently
     fails -- no joint values reach MoveIt's RobotState, planning
     stalls forever waiting on ``planning_scene_monitor`` to converge.
     We re-publish the simulator's ``/joint_states`` to a separate
     topic ``/moveit/joint_states`` with Reliable QoS; MoveIt is
     remapped to subscribe to that topic.  See
     ``moveit_launch_utils.MOVEIT_MOVE_GROUP_REMAPPINGS``.

  2. **Synthesise the prismatic ride-height joint**.  MoveIt's
     genie.urdf.xacro injects a passive ``base_footprint -> base_link``
     prismatic Z joint so RobotState can track the simulator's actual
     chassis ride height.  The simulator never publishes this joint's
     position (it isn't physical -- the chassis ride height is a
     downstream consequence of wheel contact, not a controllable DoF).
     This node looks up ``odom -> base_link`` on /tf each tick and
     appends a ``base_footprint_to_base_link`` entry to every
     republished JointState message before forwarding to MoveIt.
     Without that entry MoveIt's PlanningSceneMonitor warns "The
     complete state of the robot is not yet known.  Missing
     base_footprint_to_base_link" and FK puts base_link at z=0.

Why merged into one node: both responsibilities operate on the same
data flow (`/joint_states -> /moveit/joint_states`), and both need to
deliver every message to MoveIt at the same QoS / cadence.  Splitting
them across two processes meant MoveIt saw two independent JointState
streams arriving out of order, which RobotState's "is the state
complete" check is sensitive to.  One process, one publish per input
message, complete state in every output message.
"""

import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState
import tf2_ros


class MoveItJointStatesBridge(Node):
    """One-process bridge: SensorData /joint_states -> Reliable /moveit/joint_states
    with the synthetic ``base_footprint_to_base_link`` ride-height appended."""

    def __init__(self):
        super().__init__("moveit_joint_states_bridge")
        self.declare_parameter("input_topic", "/joint_states")
        self.declare_parameter("output_topic", "/moveit/joint_states")
        self.declare_parameter("ride_height_joint_name", "base_footprint_to_base_link")
        self.declare_parameter("ride_height_source_frame", "odom")
        self.declare_parameter("ride_height_target_frame", "base_link")

        in_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        out_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self._rh_name = self.get_parameter("ride_height_joint_name").get_parameter_value().string_value
        self._rh_source = self.get_parameter("ride_height_source_frame").get_parameter_value().string_value
        self._rh_target = self.get_parameter("ride_height_target_frame").get_parameter_value().string_value

        # Reliable QoS on the output side -- MoveIt's default for joint state
        # consumers.  Mismatched QoS would silently drop messages on the
        # subscriber side; this is the original raison d'etre of the
        # joint_states_relay layer.
        moveit_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._pub = self.create_publisher(JointState, out_topic, moveit_qos)
        self.create_subscription(JointState, in_topic, self._on_joint_states, qos_profile_sensor_data)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Cached ride-height value, updated whenever a fresh /tf lookup
        # succeeds.  Initial value 0.0 means MoveIt sees base_link at
        # ground level until the first odom->base_link transform arrives;
        # at 100 Hz publish rate from the simulator this is a few tens of
        # milliseconds at startup.
        self._rh_lock = threading.Lock()
        self._rh_z = 0.0
        self._rh_warned = False

        self.get_logger().info(
            f"moveit_joint_states_bridge: '{in_topic}' (SensorData) -> "
            f"'{out_topic}' (Reliable), with synthetic '{self._rh_name}' "
            f"= z('{self._rh_source}' -> '{self._rh_target}') appended"
        )

    def _refresh_ride_height(self):
        """Look up the current chassis ride height on /tf.

        Cheap (microseconds) and called once per inbound JointState
        message rather than on a fixed timer, so the synthetic entry's
        timestamp matches the rest of the message's data.  TF lookup
        latency is bounded by the buffer's interpolation depth, not
        wall-clock waits.
        """
        try:
            tf = self._tf_buffer.lookup_transform(self._rh_source, self._rh_target, rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException):
            if not self._rh_warned:
                self.get_logger().warn(
                    f"waiting for {self._rh_source} -> {self._rh_target} on /tf; "
                    f"ride-height defaults to 0 until available"
                )
                self._rh_warned = True
            return
        self._rh_warned = False
        with self._rh_lock:
            self._rh_z = float(tf.transform.translation.z)

    def _on_joint_states(self, msg: JointState):
        self._refresh_ride_height()
        # Don't mutate the inbound message in place -- callers may be
        # other intra-process subscribers.  Copy the fields we keep.
        out = JointState()
        out.header = msg.header
        out.name = list(msg.name) + [self._rh_name]
        out.position = list(msg.position) + [self._rh_z]
        # Velocity / effort: append zeros only when the inbound message
        # has those arrays sized to match name/position.  An empty array
        # on the input side means "not provided"; preserve that semantic.
        if len(msg.velocity) == len(msg.name):
            out.velocity = list(msg.velocity) + [0.0]
        else:
            out.velocity = list(msg.velocity)
        if len(msg.effort) == len(msg.name):
            out.effort = list(msg.effort) + [0.0]
        else:
            out.effort = list(msg.effort)
        self._pub.publish(out)


def main():
    rclpy.init()
    node = MoveItJointStatesBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
