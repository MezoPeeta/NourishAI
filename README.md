NourishAI
==============================
A food vision project made by PyTorch and with implementation of MLOps.

The project structure is inspired by [cookiecutter-data-science](https://drivendata.github.io/cookiecutter-data-science/).

Right now, MLOPs is at level 0. The model is trained on
![MLOPSLevel0](MLOPS0.jpeg)

## Packages Used
- PyTorch (for model training) & Torchvision (for data loading)
- DVC (Data Version Control) for data versioning
- WandB (Weights and Biases) for experiment tracking
- Streamlit (for deployment) soon
- Docker (for containerization) soon


The dataset used is Food-101 dataset. It is a dataset of 101 food categories, with 101,000 images. For more information, visit [here](https://www.kaggle.com/dansbecker/food-101).

The model can differentiate between food and non-food images.

Right now the model is trained on 30% of the dataset, and the accuracy is 56% on the test set.

