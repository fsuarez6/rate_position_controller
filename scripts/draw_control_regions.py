#! /usr/bin/env python
import rospy
# Messages
from visualization_msgs.msg import Marker, MarkerArray
# Math
import numpy as np

class Regions:
	def __init__(self):
		# Workspace definition
		width = self.read_parameter('~width', 140.0)
		height = self.read_parameter('~height', 100.0)
		depth = self.read_parameter('~depth', 55.0)
		self.workspace = np.array([width, height, depth])
		self.position_ratio = self.read_parameter('~position_ratio', 1000.0)		
		# Setup Subscribers/Publishers
		self.vis_pub = rospy.Publisher('visualization_marker_array', MarkerArray)
		# Initial values
		self.frame_id = '/base'
		rospy.sleep(1.0)
	
	def read_parameter(self, name, default):
		if not rospy.has_param(name):
			rospy.logwarn('Parameter [%s] not found, using default: %s' % (name, default))
		return rospy.get_param(name, default)
		
	def build_marker(self, name, position, scale, color):
		marker = Marker()
		marker.header.frame_id = self.frame_id
		marker.header.stamp = rospy.Time.now()
		marker.id = 0;
		marker.type = marker.SPHERE
		marker.ns = name
		marker.action = marker.ADD
		marker.pose.position.x = position[0]
		marker.pose.position.y = position[1]
		marker.pose.position.z = position[2]
		#~ Workspace ellipsoid: self.workspace
		marker.scale.x = scale[0]
		marker.scale.y = scale[1]
		marker.scale.z = scale[2]
		# color = [r, g, b, alpha]
		marker.color.r = color[0]
		marker.color.g = color[1]
		marker.color.b = color[2]
		marker.color.a = color[3]
		return marker
	
	def draw(self, position):
		#~ Colors
		yellow = [1.0, 1.0, 0.2, 0.6]
		green = [0.0, 0.8, 0.0, 0.25]
		#~ Position marker
		pos_scale = 2 * self.workspace / self.position_ratio
		pos_scale = pos_scale[[0,2,1]]
		pos_marker = self.build_marker('position', [0.0019956, 0.17331, 0.090887], pos_scale, yellow)
		#~ Rate marker
		rate_scale = 2.5 * self.workspace / self.position_ratio
		rate_scale = rate_scale[[0,2,1]]
		rate_marker = self.build_marker('rate', [0.0019956, 0.17331, 0.090887], rate_scale, green)
		#~ Publish
		marker_a = MarkerArray()
		marker_a.markers.append(pos_marker)
		marker_a.markers.append(rate_marker)		
		self.vis_pub.publish(marker_a)

		
if __name__ == '__main__':
	rospy.init_node('draw_control_regions')
	d = Regions()
	d.draw([0.0019948, 0.17325, 0.090974])
