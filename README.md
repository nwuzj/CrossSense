![Alt text](documentation/logo.png?raw=true "CrossSense")

# CrossSense
**CrossSense** is an open source framework for scaling up WiFi sensing to new environments and larger problems. It uses machine learning techniques to address the problem. To reduce the cost of sensing model training data collection, CrossSense employs machine learning to train, *off-line*, a roaming model to generate, from one set of measurements, synthetic training samples for each target environment. 
To scale up to a larger problem size, CrossSense adopts a mixture-of-experts approach where multiple specialized sensing models, or *experts*, are used to capture the mapping from diverse WiFi inputs to the desired outputs.

# Prerequisites
**CrossSense** is built upon the python [scikit-learn](http://scikit-learn.org) machine learning package. 

# Dataset
To obtain our dataset, please follow the instructions [here](https://github.com/nwuzj/CrossSense/blob/master/documentation/Dataset%20Release%20Agreement.pdf).

# Importance
**CrossSense** is not production ready. It's a research prototype that demonstrates the viability of applying machine learning to scale up wireless based sensing. If you encounter any problems, please file an issue on github. 

# License

Source code of CrossSense is released under the [Apache License (v2.0)](http://www.apache.org/licenses/LICENSE-2.0). 

# Citation

```bibtex
@inproceedings{crosssense,
  title={CrossSense: Towards Cross-Site and Large-Scale WiFi Sensing},
  author={Zhang, Jie and Tang, Zhanyong and Li, Meng and Fang, Dingyi and Nurmi, Petteri and Wang, Zheng},
  booktitle={The 24th ACM International Conference on Mobile Computing and Networking},
  series = {MobiCom '18},
  year={2018},
  organization={ACM}
}
```

