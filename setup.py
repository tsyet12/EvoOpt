from setuptools import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()



setup(name='EvoOpt', 
version='0.13', 
license='BSD 2-Clause',
description="EvoOpt: Python Implementation of State-of-Art Evolutionary Algorithms",
author='Sin Yong Teng',
long_description=long_description,
long_description_content_type="text/markdown",
author_email='tsyet12@gmail.com',
keywords = ['Evolutionary', 'Optimization', 'Algorithm'],
packages=find_packages(),
setup_requires=['numpy', 'matplotlib'],
classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: BSD License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
  )