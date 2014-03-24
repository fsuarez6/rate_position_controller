cmake_minimum_required(VERSION 2.8.3)
project(rate_position_controller)

find_package(catkin REQUIRED COMPONENTS 
  omni_msgs 
  grips_msgs 
  visualization_msgs
)

catkin_package(DEPENDS 
  executive_smach 
  omni_msgs 
  grips_msgs 
  visualization_msgs
)

catkin_python_setup() 

include_directories(
  ${catkin_INCLUDE_DIRS}
)
