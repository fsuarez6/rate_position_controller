cmake_minimum_required(VERSION 2.8.3)
project(rate_position_controller)

find_package(catkin REQUIRED COMPONENTS 
  baxter_core_msgs
  visualization_msgs
)

catkin_package(CATKIN_DEPENDS
  baxter_core_msgs
  omni_msgs
  smach
  smach_ros
  visualization_msgs
)

catkin_python_setup() 

include_directories(
  ${catkin_INCLUDE_DIRS}
)
