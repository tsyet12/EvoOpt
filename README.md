# EvoOpt - Evolutionary Optimization in Python

![EvoOpt Logo](https://user-images.githubusercontent.com/19692103/58713060-1de5bc00-83c2-11e9-8213-bf69e3382321.jpg)

![Build Status](https://travis-ci.com/tsyet12/EvoOpt.svg?branch=master)
![PyPI-Status](https://img.shields.io/pypi/status/EvoOpt.svg?color=blue)
![PyPI](https://img.shields.io/pypi/v/EvoOpt.svg?color=green) 
![dependencies](https://img.shields.io/librariesio/github/tsyet12/EvoOpt.svg)
![PyPI-Implementation](https://img.shields.io/pypi/implementation/EvoOpt.svg)
![Python Version](https://img.shields.io/pypi/pyversions/EvoOpt.svg)
![PyPI-License](https://img.shields.io/pypi/l/EvoOpt.svg?color=Green)

Python implementation of state-of-art meta-heuristic and evolutionary optimisation algorithms. 

** This library is implemented in Numpy (which was written in C) for fast processing speed**



# Current support for algorithms:

[x] Genetic Algorithm

[x] Duelist Algorithm

[X] Particle Swarm Optimization

[X] Gravitational Search Algorithm

[X] Firefly Algorithm

[X] Simulated Annealing

[ ] Multi-Verse Optimization

[ ] Grey-Wolf Optimization

# Quick Example

**There are four simple steps to run an optimization problem using EvoOpt**

(Example 2 from example folder)

** Prerequisites **

```python
from solver.DuelistAlgorithm import DuelistAlgorithm
```

** 1. Define your function. Say you want to minimize the equation f=(x1,x2) = (x1)^2+(x2)^2 **

```python
def f(x1,x2):
	return x1*x1+x2*x2
```

** 2. Define the variables that can be *manipulated* for optimization. Define their names as string and put them in an array. **

```python
x=["x1","x2"]
```

** 3. Define the boundaries for the manipulated variables:**

 Say:

 x1 is bounded from -2 to 10 (-2 is min value of x1 and 10 is max value of x1)

 x2 is bounded from 10 to 15 (10 is min value of x2 and 15 is max value of x2)
 
  We can arrange these boundaries according to the definition array in step 2.
  
 | Variables | x1 | x2 |
 | :---: | :---: | :---: |
 | Min | -2 | 5 |
 | Max | 10 | 15 |

The corresponding code is:

```python
 xmin=[-2,5]
 xmax=[10,15]
```

** 4. Setup the solver and start the solve procedure. **

```python
DA=DuelistAlgorithm(f,x,xmin,xmax,max_gen=1000)
DA.solve(plot=True)
```


# Results
![Result Image](https://user-images.githubusercontent.com/19692103/58713291-892f8e00-83c2-11e9-8756-e27967c32453.png)



# Dependencies
Numpy and Matplotlib

Windows:
```Bash

$python -m pip install numpy matplotlib

```

Linux:

```Bash
$pip install numpy matplotlib
```


# Installing EvoOpt

You can use two methods for installation:

**1. Install from github (recommended as this will download the newest version)**

First download the git repository. You can do this by clicking the download button or using the git command:
```BASH
$ git pull https://github.com/tsyet12/EvoOpt
```
<b>
  
Move to the directory:
  
```BASH
$ cd (directory of EvoOpt)
```

Run setup. The following command installs all files in directory:

```BASH
$ pip install -e .
```

Move to examples and run the examples

```BASH
$ cd examples
```

**1. Install from pip **

You can install this package from pip. 

Linux:

```BASH
$ pip install EvoOpt
```

Windows:

```BASH
$ python -m pip install EvoOpt
```



# More Updates to Come
