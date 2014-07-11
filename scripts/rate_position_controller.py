#! /usr/bin/env python
"""
Notes
-----
Calculations are carried out with numpy.float64 precision.

This Python implementation is not optimized for speed.

Angles are in radians unless specified otherwise.

Quaternions ix+jy+kz+w are represented as [x, y, z, w].
"""
import rospy
# Messages
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from grips_msgs.msg import GripsState
from omni_msgs.msg import OmniState, OmniFeedback, OmniButtonEvent
from geometry_msgs.msg import Vector3, Quaternion, Transform, PoseStamped
from visualization_msgs.msg import Marker
# State Machine
import smach
import smach_ros
from smach import CBState
# Math
from math import pi, exp, sin, sqrt
import numpy as np
import tf.transformations as tr

class TextColors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'

  def disable(self):
    self.HEADER = ''
    self.OKBLUE = ''
    self.OKGREEN = ''
    self.WARNING = ''
    self.FAIL = ''
    self.ENDC = ''

class RatePositionController:
  STATES = ['GO_TO_CENTER', 'POSITION_CONTROL', 'VIBRATORY_PHASE', 'RATE_CONTROL', 'RATE_COLLISION']
  def __init__(self):
    # Create a SMACH state machine
    self.sm = smach.StateMachine(outcomes=['succeeded', 'aborted'])
    with self.sm:
      # Add states to the state machine
      smach.StateMachine.add('GO_TO_CENTER', CBState(self.go_to_center, cb_args=[self]), 
                             transitions={'lock': 'GO_TO_CENTER', 'succeeded': 'POSITION_CONTROL', 'aborted': 'aborted'})
      smach.StateMachine.add('POSITION_CONTROL', CBState(self.position_control, cb_args=[self]), 
                             transitions={'stay': 'POSITION_CONTROL', 'leave': 'VIBRATORY_PHASE', 'aborted': 'aborted'})
      smach.StateMachine.add('VIBRATORY_PHASE', CBState(self.vibratory_phase, cb_args=[self]),
                             transitions={'vibrate': 'VIBRATORY_PHASE', 'succeeded': 'RATE_CONTROL', 'aborted': 'aborted'})
      smach.StateMachine.add('RATE_CONTROL', CBState(self.rate_control, cb_args=[self]), 
                             transitions={'stay': 'RATE_CONTROL', 'leave': 'GO_TO_CENTER', 'collision': 'RATE_COLLISION', 'aborted': 'aborted'})
      smach.StateMachine.add('RATE_COLLISION', CBState(self.rate_collision, cb_args=[self]),
                             transitions={'succeeded': 'GO_TO_CENTER', 'aborted': 'aborted'})
    
    # Read all the parameters from the parameter server
    # Topics to interact
    master_name = self.read_parameter('~master_name', 'phantom')
    slave_name = self.read_parameter('~slave_name', 'grips')
    self.master_state_topic = '/%s/state' % master_name
    self.feedback_topic = '/%s/force_feedback' % master_name
    self.slave_state_topic = '/%s/state' % slave_name
    self.ik_mc_topic = '/%s/ik_command' % slave_name
    self.gripper_topic = '/%s/GRIP/command' % slave_name
    # Workspace definition
    self.units = self.read_parameter('~units', 'mm')
    width = self.read_parameter('~workspace/width', 140.0)
    height = self.read_parameter('~workspace/height', 100.0)
    depth = self.read_parameter('~workspace/depth', 55.0)
    self.center_pos = self.read_parameter('~workspace/center', [0, 0 ,0])
    self.workspace = np.array([width, depth, height])
    self.hysteresis = self.read_parameter('~hysteresis', 3.0)
    self.pivot_dist = self.read_parameter('~pivot_dist', 5.0)
    # Force feedback parameters
    self.k_center = self.read_parameter('~k_center', 0.1)
    self.b_center = self.read_parameter('~b_center', 0.003)
    self.k_rate = self.read_parameter('~k_rate', 0.05)
    self.b_rate = self.read_parameter('~b_rate', 0.003)
    # Position parameters
    self.rpy_offset = self.read_parameter('~rpy_offset', [0,0,0])
    self.q_offset = tr.quaternion_from_euler(*self.rpy_offset)
    self.publish_frequency = self.read_parameter('~publish_rate', 1000.0)
    self.position_ratio = self.read_parameter('~position_ratio', 250)
    # Vibration parameters
    self.vib_a = self.read_parameter('~vibration/a', 2.0)             # Amplitude (mm)
    self.vib_c = self.read_parameter('~vibration/c', 5.0)             # Damping
    self.vib_freq = self.read_parameter('~vibration/frequency', 30.0) # Frequency (Hz)
    self.vib_time = self.read_parameter('~vibration/duration', 0.3)   # Duration (s)
    self.vib_start_time = 0.0
    # Rate parameters
    self.rate_pivot = np.zeros(3)
    self.rate_gain = self.read_parameter('~rate_gain', 1.0) 
    
    # Setup Subscribers/Publishers
    rospy.Subscriber(self.master_state_topic, OmniState, self.cb_master_state)
    rospy.Subscriber(self.slave_state_topic, GripsState, self.cb_slave_state)
    self.feedback_pub = rospy.Publisher(self.feedback_topic, OmniFeedback)
    self.ik_mc_pub = rospy.Publisher(self.ik_mc_topic, PoseStamped)
    self.gripper_pub = rospy.Publisher(self.gripper_topic, Float64)
    self.vis_pub = rospy.Publisher('visualization_marker', Marker)
    # Initial values
    self.frame_id = self.read_parameter('~reference_frame', 'world')
    self.colors = TextColors()
    self.gripper_cmd = 0.0
    self.master_pos = None
    self.master_rot = np.array([0, 0, 0, 1])
    self.master_vel = np.zeros(3)
    self.master_dir = np.zeros(3)
    self.slave_pos = None
    self.slave_rot = np.array([0, 0, 0, 1])
    self.timer = None
    self.force_feedback = np.zeros(3)
    # Slave command
    self.slave_synch_pos = np.zeros(3)
    
    self.loginfo('Waiting for [%s] and [%s] topics' % (self.master_state_topic, self.slave_state_topic))
    while not rospy.is_shutdown():
      if (self.slave_pos == None) or (self.master_pos == None):
        rospy.sleep(0.01)
      else:
        self.loginfo('Rate position controller running')
        # Register rospy shutdown hook
        rospy.on_shutdown(self.shutdown_hook)
        break
    
    # Make sure the first command sent to the slave is equal to its current position6D
    self.command_pos = np.array(self.slave_pos)
    self.command_rot = np.array(self.slave_rot)
    
    # Start the timer that will publish the ik commands
    self.timer = rospy.Timer(rospy.Duration(1.0/self.publish_frequency), self.publish_command)
    
    self.loginfo('State machine state: GO_TO_CENTER')
    
  @smach.cb_interface(outcomes=['lock', 'succeeded', 'aborted'])
  def go_to_center(user_data, self):
    if not np.allclose(np.zeros(3), self.master_pos, atol=self.hysteresis):
      self.force_feedback = (self.k_center * self.master_pos + self.b_center * self.master_vel) * -1.0
      return 'lock'
    else:
      self.force_feedback = np.zeros(3)
      self.slave_synch_pos = np.array(self.slave_pos)
      self.command_pos = np.array(self.slave_pos)
      self.command_rot = np.array(self.slave_rot)
      self.draw_position_region(self.slave_synch_pos)
      self.loginfo('State machine transitioning: GO_TO_CENTER:succeeded-->POSITION_CONTROL')
      return 'succeeded'
  
  @smach.cb_interface(outcomes=['stay', 'leave', 'aborted'])
  def position_control(user_data, self):
    if self.inside_workspace(self.master_pos):
      self.command_pos = self.slave_synch_pos + self.master_pos / self.position_ratio
      self.command_rot = np.array(self.master_rot)
      return 'stay'
    else:
      self.command_pos = np.array(self.slave_pos)
      self.command_rot = np.array(self.slave_rot)
      self.vib_start_time = rospy.get_time()
      self.loginfo('State machine transitioning: POSITION_CONTROL:leave-->VIBRATORY_PHASE')
      return 'leave'
    
  @smach.cb_interface(outcomes=['vibrate', 'succeeded', 'aborted'])
  def vibratory_phase(user_data, self):
    if rospy.get_time() < self.vib_start_time + self.vib_time:
      t = rospy.get_time() - self.vib_start_time 
      amplitude = -self.vib_a*exp(-self.vib_c*t)*sin(2*pi*self.vib_freq*t);
      self.force_feedback = amplitude * self.master_dir
      return 'vibrate'
    else:
      # The pivot point should be inside the position area but it's better when we use the center
      #~ self.rate_pivot = self.master_pos - self.pivot_dist * self.normalize_vector(self.master_pos)
      self.force_feedback = np.zeros(3)
      self.rate_pivot = np.array(self.master_pos)
      self.loginfo('State machine transitioning: VIBRATORY_PHASE:succeeded-->RATE_CONTROL')
      return 'succeeded'
  
  @smach.cb_interface(outcomes=['stay', 'leave', 'collision', 'aborted'])
  def rate_control(user_data, self):
    if not self.inside_workspace(self.master_pos):
      # Send the force feedback to the master
      self.force_feedback = (self.k_rate * self.master_pos + self.b_rate * self.master_vel) * -1.0
      # Send the rate command to the slave
      distance = sqrt(np.sum((self.master_pos - self.rate_pivot) ** 2)) / self.position_ratio
      self.command_pos += (self.rate_gain * distance * self.normalize_vector(self.master_pos)) / self.position_ratio
      self.command_rot = np.array(self.master_rot)
      return 'stay'
    else:
      self.command_pos = np.array(self.slave_pos)
      self.command_rot = np.array(self.slave_rot)
      self.force_feedback = np.zeros(3)
      self.loginfo('State machine transitioning: RATE_CONTROL:leave-->GO_TO_CENTER')
      return 'leave'
    
  @smach.cb_interface(outcomes=['succeeded', 'aborted'])
  def rate_collision(user_data, self):
    return 'succeeded'
  
  def execute(self):
    self.sm.execute()
      
  def shutdown_hook(self):
    # Stop the state machine
    self.sm.request_preempt()
    # Stop the publisher timer
    self.timer.shutdown()   
  
  def read_parameter(self, name, default):
    if not rospy.has_param(name):
      rospy.logwarn('Parameter [%s] not found, using default: %s' % (name, default))
    return rospy.get_param(name, default)
  
  def loginfo(self, msg):
    rospy.logwarn(self.colors.OKBLUE + str(msg) + self.colors.ENDC)
  
  def inside_workspace(self, point):
    # The workspace as an ellipsoid: http://en.wikipedia.org/wiki/Ellipsoid
    return np.sum(np.divide(point**2, self.workspace**2)) < 1
    
  def normalize_vector(self, v):
    result = np.array(v)
    result /= np.sqrt(np.sum((result ** 2)))
    return result
  
  def send_feedback(self):
    feedback_msg = OmniFeedback()
    feedback_msg.force.x = self.force_feedback[0]
    feedback_msg.force.y = self.force_feedback[1]
    feedback_msg.force.z = self.force_feedback[2]
    feedback_msg.position.x = self.center_pos[0]
    feedback_msg.position.y = self.center_pos[1]
    feedback_msg.position.z = self.center_pos[2]
    self.feedback_pub.publish(feedback_msg)
  
  # DO NOT print to the console within this function
  def cb_master_state(self, msg):    
    self.master_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]) - self.center_pos
    self.master_rot = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
    self.master_rot = tr.quaternion_multiply(self.q_offset, self.master_rot)
    self.master_vel = np.array([msg.velocity.x, msg.velocity.y, msg.velocity.z])
    self.master_dir = self.normalize_vector(self.master_vel)
    # Control the gripper
    cmd = self.gripper_cmd
    if msg.close_gripper:
      cmd = 5.0
    else:
      cmd = -5.0
    try:
      self.gripper_pub.publish(cmd)
    except rospy.exceptions.ROSException:
      pass
  
  def cb_slave_state(self, msg):
    self.slave_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    self.slave_rot = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
  
  def publish_command(self, event):
    position, orientation = self.command_pos, self.command_rot
    ik_mc_msg = PoseStamped()
    ik_mc_msg.header.frame_id = self.frame_id
    ik_mc_msg.header.stamp = rospy.Time.now()
    ik_mc_msg.pose.position.x = position[0]
    ik_mc_msg.pose.position.y = position[1]
    ik_mc_msg.pose.position.z = position[2]
    ik_mc_msg.pose.orientation.x = orientation[0]
    ik_mc_msg.pose.orientation.y = orientation[1]
    ik_mc_msg.pose.orientation.z = orientation[2]
    ik_mc_msg.pose.orientation.w = orientation[3]
    try:
      self.ik_mc_pub.publish(ik_mc_msg)
      self.send_feedback()
    except rospy.exceptions.ROSException:
      pass
  
  def draw_position_region(self, center_pos):
    marker = Marker()
    marker.header.frame_id = self.frame_id
    marker.header.stamp = rospy.Time.now()
    marker.id = 0;
    marker.type = marker.SPHERE
    marker.ns = 'position_region'
    marker.action = marker.ADD
    marker.pose.position.x = center_pos[0]
    marker.pose.position.y = center_pos[1]
    marker.pose.position.z = center_pos[2]
    #~ Workspace ellipsoid: self.workspace
    marker.scale.x = 2 * self.workspace[0]/self.position_ratio
    marker.scale.y = 2 * self.workspace[1]/self.position_ratio
    marker.scale.z = 2 * self.workspace[2]/self.position_ratio
    marker.color.a = 0.5
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 0.2
    #~ Publish
    self.vis_pub.publish(marker)

    
if __name__ == '__main__':
  rospy.init_node('rate_position_controller', log_level=rospy.WARN)
  try:
    controller = RatePositionController()
    controller.execute()
  except rospy.exceptions.ROSInterruptException:
      pass
