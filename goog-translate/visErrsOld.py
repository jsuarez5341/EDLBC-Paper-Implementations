import sys
from matplotlib import pyplot as plt
import numpy as np
import time

if __name__ == '__main__':
   root = sys.argv[1]

   train       = np.load(root + 'npTrain.npy')
   val         = np.load(root + 'npVal.npy')
   trainBleu = np.load(root + 'npTrainBleu.npy')
   valBleu   = np.load(root + 'npValBleu.npy')

   T = np.arange(len(train)) * 25

   plt.hold(True)
   plt.plot(T, train, 'b', linewidth=3)
   plt.plot(T, val, 'g', linewidth=3)
   plt.plot(T, trainBleu, 'r', linewidth=3)
   plt.plot(T, valBleu, 'k', linewidth=3)
   plt.xlabel('Minibatch Iterations')
   plt.ylabel('Cross Entropy Loss')

   plt.show()

   

