#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
          packages=['rate_position_controller'],
              package_dir={'': 'scripts'})

setup(**setup_args)
