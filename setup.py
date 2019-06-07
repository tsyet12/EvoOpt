from setuptools import setup, find_packages

setup(name='EvoOpt', 
version='0.1', 
license='BSD 2-Clause',
description="EvoOpt: Python Implementation of State-of-Art Evolutionary Algorithms",
author='Sin Yong Teng',
author_email='tsyet12@gmail.com',
download_url = 'https://github.com/tsyet12/EvoOpt',
keywords = ['Evolutionary', 'Optimization', 'Algorithm'],
packages=find_packages(),
setup_requires=['numpy', 'matplotlib'],
classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: BSD 2-Clause',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
  )