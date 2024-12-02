# PI-GANO

This repository is the official implementation of the paper: [Physics-Informed Geometry-Aware Neural Operator](https://www.sciencedirect.com/science/article/pii/S0045782524007941?via%3Dihub), published in Journal of Computer Methods in Applied Mechanics and Engineering. The arxiv version of paper can also be found [here](https://arxiv.org/html/2408.01600v1).

Our research explores physics-informed machine learning methods for **variable domain geometry** where PDE solution domains are represented as a set of collocation point coordinates, making the approach **geometry-aware**. In this paper, we introduce the first neural operator model that is geometry-aware and can be trained without any FEM data needed. However, the model architecture we propose in this paper is intentionally kept straightforward, and we encourage researchers to explore and develop more advanced architectures to further enhance this approach. 

## Overview

If you're interested in using our well-trained model, please refer to the **"User mode"** section. For those with similar research interests looking to explore more advanced model architectures and training algorithms, please check the **"Developer mode"** section. This work is also one of our work for developing [**Neural Operators as Foundation Model for solving PDEs**](https://github.com/WeihengZ/Physics-informed-Neural-Foundation-Operator). Please feel free to check it out as well if you are interested! We are excited to see more and more interesting ideas coming out for this research goal!

## User mode

If you want to reproduce the results of our paper, please first download our [dataset](https://drive.google.com/drive/folders/1ZcKAMCESzhQZXNjbxItKRlpISjAhT2hI?usp=sharing) into the folder named "data", and download our [well-trained models](https://drive.google.com/drive/folders/1n9ens6nK_-QcidqLZq1Pq_wkLo-TPkzu?usp=sharing) into the folder named "res/saved_models". 

After preparing all the data and well-trained models, you need to first install all the required python package (with the Python>=3.8 is preferred) by
```
pip install -r requirements.txt
```

Then you can evaluate the prediction accuracy of our proposed GANO model in the testing dataset for the darcy problem and 2D plate stress problem with the following commands:
```
python PINO_darcy_training.py --model='GANO' --phase='test'
python PINO_plate_training.py --model='GANO' --phase='test'
```

Or implement the model training by replacing the "phase" argument:
```
python PINO_darcy_training.py --model='GANO' --phase='train'
python PINO_plate_training.py --model='GANO' --phase='train'
```

If you want to reproduce the results of the baseline model: Physics-informed Mesh-independent Deep Compositional Operator Network, you can simply replace the "model" argument:
```
python PINO_darcy_training.py --model='DCON' --phase='train'
python PINO_plate_training.py --model='DCON' --phase='train'
```

## Developer mode

This repository is user-friendly for developing new model architecture of the neural operator model. You can simply explore your self-designed model architecture by the following steps:
* open the script of the model_darcy.py or model_plate.py, input your self-defined model architectur into the function class of "New_model_darcy" and "New_model_plate".
* Add the hyper-parameters in the file "configs/self_defined_Darcy_star.yaml" and "configs/self_defined_plate_stress_DG.yaml". 
* Train your model by simply run the following command:
```
cd Main
python PINO_darcy_training.py --model='self_defined' --phase='train'
python PINO_plate_training.py --model='self_defined' --phase='train'
```

If you are interested in developing more advanced training algorithms, please check our the script "utils_darcy_train.py" and "utils_plate_train.py".

**If you think that the work of the PI-GANO is useful in your research, please consider citing our paper in your manuscript:**
```
@article{zhong2025physics,
  title={Physics-Informed Geometry-Aware Neural Operator},
  author={Zhong, Weiheng and Meidani, Hadi},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={434},
  pages={117540},
  year={2025},
  publisher={Elsevier}
}
```
