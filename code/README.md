# Modular Addition

Final Project for Mathematical Introduction to Machine Learning.

Gives a simple implementation of Transformer model, as a recurrence to Grokking phenomenon according to
paper [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177).

## Requirements

You need GPU which supports [TensorFloat32](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/)(TF32) floating point format to reproduce some results.

Our code is implemented in Python and fully tested on Python 3.13.1 and 3.12.7 with PyTorch 2.5.1.
All the required packages are contained in `requirements.txt`. Install them by running the following command:

```bash
pip install -r requirements.txt
```

## Usage

To train the model with different configurations, run the following command:

```shell
cd code/src
python main.py -p <configuration_path>
```

The path is optional, and use the default configuration if not given. We give our configurations in `configs` directory. Take the Transformer model as an example:

```shell
python main.py -p configs/transformer/Adam.jsonc
```

DIY your own configuration file by following the format in `configs/template.jsonc`.

## Structure

Core files in the repository are organized as follows:

```
code/src
├── main.py
├── train.py
├── plot.py
└── modular_add
    ├── __init__.py 
    ├── data.py
    ├── model.py
    ├── optim.py
    ├── params.py
    └── util.py
```

- `main.py`: Entry point of the program, which parses the command line parameters and starts the training process.
- `train.py`: Training loop and evaluation process.
- `plot.py`: Script to plot the results.
- `modular_add`: Package containing the core implementation of the models.
  - `data.py`: Generation of dataset and data loader.
  - `model.py`: implementation of the Transformer/LSTM/MLP model.
  - `optim.py`: Optimizer and learning rate scheduler.
  - `params.py`: Data class to store the hyperparameters.
  - `util.py`: Utility functions.

## Documentation

See the [report](./doc/main.tex) for more details.
