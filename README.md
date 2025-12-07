Smooth Neural Network Visualizer (Pygame)

A simple interactive neural network visualizer built with pure Python and Pygame. Trains a small ANN on the Wine Classification dataset (13 → 6 → 4 → 3) and shows smooth activations, animated weights, particles, and live loss/accuracy graphs.

Features

Smooth, interpolated activation transitions

Animated weight visualization and node color changes

Particle effects to show data flow during forward passes

Live loss and accuracy charts

UI buttons to start/pause and toggle weights/particles

Requirements

pip install pygame torch scikit-learn numpy


Run

python nn_visualizer_smooth.py


Dataset
Uses the Wine Classification dataset from sklearn.datasets.

Files

nn_visualizer_smooth.py — main script containing model, trainer, and visualizer

Purpose
Educational demo for visualizing how a feedforward neural network trains and how activations and weights evolve over time.


<img width="2844" height="1615" alt="image" src="https://github.com/user-attachments/assets/8af0b75b-0342-4555-8afc-952365dd9ca9" />
