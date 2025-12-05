from Class_perceptron import Perceptron
import numpy as np

x, y = Perceptron.generate_LS(n=2, p=20)

perceptron_to_verify = Perceptron()

perceptron_to_verify.train_with_batch(x, y)
