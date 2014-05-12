#! /usr/bin/env python
import rospy, time, math
# Sockets
import socket, struct
# Messages
from StringIO import StringIO
from geometry_msgs.msg import PoseStamped
from omni_msgs.msg import OmniState, OmniFeedback, OmniButtonEvent
from labview_bridge import LabviewServer

class HardwareInterface(LabviewServer):
  def __init__(self): 
    LabviewServer.__init__(self)
    # Load parameters
    self.device_name = self.read_parameter('~device_name', 'haptic_device')
    self.reference_frame = self.read_parameter('~reference_frame', 'world')
    self.units = self.read_parameter('~units', 'mm')
    # Topics
    self.state_topic = '/%s/state' % self.device_name
    self.feedback_topic = '/%s/force_feedback' % self.device_name
    self.pose_topic = '/%s/pose' % self.device_name
    # Setup Subscribers/Publishers
    rospy.Subscriber(self.feedback_topic, OmniFeedback, self.cb_feedback)
    self.state_pub = rospy.Publisher(self.state_topic, OmniState)
    self.pose_pub = rospy.Publisher(self.pose_topic, PoseStamped)
  
  def execute(self):
    while not rospy.is_shutdown():
      pose_msg = PoseStamped()
      recv_msg = OmniState()
      data = self.recv_timeout()
      if data:
        # Serialize received UDP message
        recv_msg.deserialize(data)
        recv_msg.header.stamp = rospy.Time.now()
        # Publish device state
        self.state_pub.publish(recv_msg)
        # Publish device pose
        pose_msg.header = recv_msg.header
        pose_msg.pose = recv_msg.pose
        pose_msg.pose.position.x /= 1000.0
        pose_msg.pose.position.y /= 1000.0
        pose_msg.pose.position.z /= 1000.0
        self.pose_pub.publish(pose_msg)
        
  def cb_feedback(self, msg):
    # Serialize cmd_msg
    file_str = StringIO()
    msg.serialize(file_str)
    # Send over udp the feedback omni_msgs/OmniFeedback
    self.write_socket.sendto(file_str.getvalue(), (self.write_ip, self.write_port))


if __name__ == '__main__':
  rospy.init_node('haptic_device_interface')
  interface = HardwareInterface()
  interface.execute()
