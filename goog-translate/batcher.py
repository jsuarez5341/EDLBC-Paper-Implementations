import numpy as np
from pdb import set_trace as t

class Batcher():
   def __init__(self, X, Y):
      #Randomly shuffle data
      inds = np.arange(X.shape[0])
      permInds = np.random.permutation(inds)
      self.X = X[permInds]
      self.Y = Y[permInds]
      self.ind = 0

   def next(self, batchSize, movePointer=True):
      #Main batcher
      m = self.X.shape[0]
      ind = self.ind

      #Misses up to one batch. Don't care.
      if m - ind < batchSize:
         ind = 0
         self.ind = 0
      
      nextInd = ind + batchSize
      XRet = self.X[ind:nextInd][::-1]
      YRet = self.Y[ind:nextInd] 
      lengths = np.argmax(YRet==1, 1)+1
      ret = [XRet, YRet, lengths]
      if movePointer:
         self.ind = nextInd
      
      return ret


