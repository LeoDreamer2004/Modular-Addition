# Modular Addition

Final Project for Mathematical Introduction to Machine Learning.

Gives a simple implementation of Transformer model, as a recurrence to Grokking phenomenon according to
paper [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177).

## Software Requirements

Our code is written in Python and is fully tested on Python 3.13.1 and 3.12.7 with PyTorch 2.5.1.
File `requirements.txt` contains all the required packages to run the code. You can install them by running the following command:

```bash
pip install -r requirements.txt
```

## Hardware Requirements

You need GPU which supports [TensorFloat32](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/)(TF32) floating point format to reproduce some results.
