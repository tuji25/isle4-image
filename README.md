# Ireco:Three-layered neural networks package
Ireco is a Python package that provides three-layered neural networks.

This library designed for [Computer Science Laboratory and Exercise 4](https://ocw.kyoto-u.ac.jp/syllabus/?act=detail&syllabus_id=tech_5475&year=2022). Read documents in `docs/` for the details of assignments of the course. 

## Required library
You need [Numpy](https://numpy.org/) library to use this package.

## How to use

### Training
```
# X_train, Y_train are Numpy arrays.
neural.fit(X_train, Y_train, num_hidden_units=512, batch_size=100, epochs=20, 
           optimizer="sgd", activation="relu", filename="weight.npz")
```
The weights of the neural network are saved in `filename` after training is over.

You can select two optimizers, SGD(`"sgd"`) or AdaGrad(`"adagrad"`). 

This package implements two activation functions, sigmoid(`"sigmoid"`) and ReLU(`"relu"`).

`tests/kadaiA4.py` trains a neural network on MNIST dataset.

### Predicting
```
# X_test is a Numpy array and filename is a file of trained weight data.
prediction = neural.fpredict(X_test, filename="weight.npz", activation="relu")
```