# Detection of defects in X-Ray images using Neural Networks - MURA Dataset

*Author: Matúš Botek*


* The code in notebooks and python scripts is documented and used functions are described.
* Jupyter notebooks are used for training or visualizing specific tasks.
* The python scripts ***utils.py and image_preprocessing.py*** containt methods useful for more transparent code


Project file structure:
.
├── logs # Logged runs for cross-validation and hyperparameter tuning for both models
│   ├── DenseNet169
│   │   ├── crossval.log
│   │   └── tuner_summary.log
│   └── ResNet50
│       ├── crossval.log
│       └── tuner_summary.log
├── weights # model weights for final models trained on full dataset using cross-validations
│   ├── DenseNet169_crossval_best.h5
│   └── ResNet50_crossval_best.h5
├── src # Implementation source codes
    ├── dataset_preparation.ipynb
    ├── grad_cam.ipynb
    ├── hyperparameter_optimization_DenseNet169.ipynb
    ├── hyperparameter_optimization_ResNet50.ipynb
    ├── image_preprocessing_examples.ipynb
    ├── image_preprocessing.py
    ├── offline_image_preprocessing.ipynb
    ├── training.ipynb
    ├── utils.py
    └── visualize_augmentation.ipynb

