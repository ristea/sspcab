#  Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection (official)                                                                                  

We propose to integrate the reconstruction-based functionality into a novel self-supervised predictive architectural building block. 
The proposed self-supervised block is generic and can easily be incorporated into various state-of-the-art anomaly detection methods.

The original paper could be found at: https://arxiv.org/pdf/2111.09099.pdf

This code is released under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

-----------------------------------------

![map](resources/sspcab_all.png)

-----------------------------------------                                                                                                                                      
## Information

Our kernel is illustrated in the below picture.  The visible area of the receptive field is denoted by the regions Ki, ∀i ∈ {1, 2, 3, 4},
while the masked area is denoted by M. A dilation factor d controls the local or global nature of the visible information with respect to M.
#![map](resources/masked_kernel.png)


## Implementation

We provide implementation for both PyTorch and Tensorflow in the ``sspcab_torch.py`` and ``sspcab_tf.py``

> In order to work properly you need to have a python version newer than 3.6
>> We used the python 3.6.8 version.


## Cite us
```
@article{Ristea-CVPR-2022,
  title={Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection},
  author={Ristea, Nicolae-Catalin and Madan, Neelu and Ionescu, Radu Tudor and Nasrollahi, Kamal and Khan, Fahad Shahbaz and Moeslund, Thomas B and Shah, Mubarak},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```
## You can send your questions or suggestions to: 
r.catalin196@yahoo.ro, raducu.ionescu@gmail.com

### Last Update:
March 18, 2022 


