---
title: 'Implementation of ComPass score'
date: 2020-03-16
permalink: /posts/compass
tags:
  - python3
  - python
  - interaction
  - proteomics
  - statistics
---

For a project I was working on I needed to compare several common interaction scoring algorithms (Saint, Mist, ComPass). As the final goal was to employ them for a larger project, I decided to re-implement some of them in Python.
So in this series of posts we will be looking at how to translate either R into Python or directly from the publication method.\n

All of this scripts are available at https://github.com/fossatiA/pyAPMS

#### ComPass

Developed by the [Harper lab](https://harper.hms.harvard.edu) this scoring system considers both frequency of bait-prey interaction across different experiments and also the prey intensity.
Luckily for me there is a detailed explanation [here](http://besra.hms.harvard.edu/ipmsmsdbs/cgi-bin/tutorial.cgi)\n

The input seems to be an adjacency matrix in the form of

|                | Bait1          | Bait2          |
| :------------- | :------------- | :------------- |
| Prey1          | Prey1Bait1     | Prey1Bait2     |
| Prey2          | Prey1Bait1     | Prey2Bait2     |
| Prey3          | Prey3Bait1     | Prey3Bait2     |
| Prey4          | Prey4Bait1     | Prey4Bait2     |

In this case multiple baits are combined by taking the maximum of the spectral count across the replicates. So let's start coding!\n


#### Import all packages and generate some dummy data

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


np.random.seed(0)
distr = np.random.randint(1,50,100)
distr[np.random.choice(distr, 10)] = 0
distr = dist.reshape(10,5)
```

First we calculate the [Z-score](https://en.wikipedia.org/wiki/Standard_score) which basically represents the difference from the mean.

```
std_sc = lambda x: x - np.mean(x)/np.std(x)
zsc = np.array([std_sc(distr[i]) for i in range(distr.shape[0])])
```



```
std_sc = lambda x: x - np.mean(x)/np.std(x)
zsc = np.array([std_sc(distr[i]) for i in range(distr.shape[0])])
```
