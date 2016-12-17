import sys
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import time
from pdb import set_trace as tt

if __name__ == '__main__':
   root = sys.argv[1]
   title = sys.argv[2]

   train       = np.load(root + 'npTrain.npy')[:990]
   val         = np.load(root + 'npVal.npy')[:990]
   trainBleu = np.load(root + 'npTrainBleu.npy')[:990]
   valBleu   = np.load(root + 'npValBleu.npy')[:990]

   T = np.arange(len(train)) * 25

   fig, ax = plt.subplots()
   ls = 4
   ax.plot(T, train, 'b', linewidth=ls, label='Train Error')
   ax.plot(T, val, 'g', linewidth=ls, label='Validation Error')
   ax.plot(T, trainBleu, 'r', linewidth=ls, label='Train BLEU')
   ax.plot(T, valBleu, 'k', linewidth=ls, label='Validation BLEU')

   for item in ax.get_xticklabels() + ax.get_yticklabels():
      item.set_fontsize(20)

   fs = 25
   plt.xlabel('Minibatch Iterations', fontsize=fs)
   plt.ylabel('Cross Entropy, BLEU Score', fontsize=fs)
   plt.title(title, fontsize=fs)

   legend = ax.legend(loc='upper center', shadow=False)
   frame = legend.get_frame()
   frame.set_facecolor('0.90')

   # Set the fontsize
   for label in legend.get_texts():
      label.set_fontsize('large')

   for label in legend.get_lines():
      label.set_linewidth(6)  # the legend line width


   plt.show()

   

