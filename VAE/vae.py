import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from pdb import set_trace as t
import numpy as np
from batcher import Batcher


#Tensorflow setup
def tfInit():
   init = tf.initialize_all_variables()
   sess = tf.Session()
   sess.run(init)
   return sess

#Load MNIST
def loadDat():
   mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

   #Train
   trainDat = mnist.train

   #Val
   valDat  = mnist.validation

   #Test
   testDat = mnist.test.images

   #Dimensionality of data and num classes
   #D = trainDat.shape[1]
   D=784

   return trainDat, valDat, testDat, D

#KL(q(z|x)||p(z))
def KLDiv(mu, logsig):
   return tf.reduce_mean(-0.5*tf.reduce_sum(1 + tf.log(tf.square(logsig)) - mu**2 - logsig**2, 1))

def weightedGaussian(mu, sigma):
   return mu + tf.exp(sigma) * tf.random_normal([batchSize, autoDim], mean=0, stddev=1)

def CELoss(a, y, off=1e-4):
   a = tf.clip_by_value(a, off, 1-off)
   return tf.reduce_mean(-tf.reduce_sum(y*tf.log(a) + (1-y)*tf.log(1-a), 1))
 
def encode():
   a = x
   for l in range(auto-1):
      a = tf.nn.elu(tf.matmul(a, W[l]))
   a = tf.matmul(a, W[auto-1])
   mu = a[:, :(nil[auto]/2)] 
   logsig = a[:, (nil[auto]/2):] 
   z = weightedGaussian(mu, logsig)
   return z, mu, logsig

def decode(x):
   a = x
   for l in range(auto, len(W)-1):
      a = tf.nn.elu(tf.matmul(a, W[l]))
   return tf.nn.sigmoid(tf.matmul(a, W[-1]))

def pred():
   z, mu, logsig = encode()
   a = decode(z)
   return a, mu, logsig

#Network cost
def loss():
   a, mu, sig = pred()
   CE = CELoss(a, y)
   KL = KLDiv(mu, sig)
   #sumSq = tf.reduce_mean(tf.reduce_sum((a-y)**2, 1))
   return tf.reduce_mean(CE + KL)

def loss2():
   a, mu, sig = pred()
   CE = CELoss(a, y)
   KL = KLDiv(mu, sig)
   sumSq = tf.reduce_mean(tf.reduce_sum((a-y)**2, 1))
   return sumSq, KL, CE


#Generic gradient based optimizer
def optimizer(eta=0.5):
   return tf.train.AdamOptimizer(eta).minimize(loss())
   #return tf.train.AdagradOptimizer(.01).minimize(loss())

def train(trainDat, valDat):
   for i in range(maxIters):
      if i % epoch == 0:
         xTrain, yTrain = trainDat.next(valBatchSize)
         acc = test(xTrain, yTrain)
         print 'Training accuracy at iteration ' + str(i) + ': ' + str(acc)
         xVal, yVal = valDat.next(valBatchSize)
         acc = test(xVal, yVal)
         print 'Validation accuracy at iteration ' + str(i) + ': ' + str(acc)

         sumsq, KL, CE = sess.run(loss2(), feed_dict={x:xTrain, y:yTrain})
         print 'KL: '+str(KL)+', SumSQ: '+str(sumsq), ', CE: ' + str(CE)

      if i % (10*epoch) == 0:
         print 'Saving a sample'
         saveSample(xTrain, yTrain, i)
         saver.save(sess, 'samps/save'+str(i))
         saveManifold(25, i)
         
      #Minibatch inputs
      xBatch, yBatch = trainDat.next(batchSize)
      grads = sess.run(opt, feed_dict={x:xBatch, y:yBatch})

def test(X, Y):
   return sess.run(loss(), feed_dict={x:X, y:Y})

def evalModel(batcher):
   cost = 0
   n = 100
   for i in range(n):
      testX, testY = batcher.next(batchSize)
      cost += sess.run(loss(), feed_dict={x:testX, y:testY})
   return cost / n

def saveSample(testX, testY, i):
   a, mu, sig = pred()
   aNp = sess.run(a, feed_dict={x:testX, y:testY})
   nn = 5
   dd = 28
   imgSave = aNp[:nn**2]
   final = np.zeros((dd*nn, dd*nn))
   finalLbl = np.zeros((dd*nn, dd*nn))
   ind = -1 
   for r in range(nn):
      for c in range(nn):
         ind += 1
         imgI = imgSave[ind].reshape(dd,dd)
         final[r*dd:(r+1)*dd, c*dd:(c+1)*dd] = imgI
         imgI = testY[ind].reshape(dd,dd)
         finalLbl[r*dd:(r+1)*dd, c*dd:(c+1)*dd] = imgI
      
   final -= np.min(final)
   finalLbl -= np.min(finalLbl)
   final = (255*(final / np.max(final))).astype(np.uint8)
   finalLbl = (255*(finalLbl / np.max(finalLbl))).astype(np.uint8)
   final = np.hstack((final, finalLbl))
   np.save('samps/samp'+str(i)+'.npy', final)
   
def saveManifold(n, name):
   dd = 28
   img = np.zeros((dd*n, dd*n))
   dist = np.linspace(-5,5,n).astype(np.float32)
   dist = space.astype(np.float32)
   inpVec = np.zeros((0, 2), dtype=np.float32)
   for r in range(n) :
      zr = dist[r] 
      for c in range(n):
         zc = dist[c]
         inpVec = np.vstack((inpVec, np.array([[zr, zc]])))
   imgI = sess.run(decode(inpVec), feed_dict={})
   i = -1
   for r in range(n):
      for c in range(n):
         i += 1
         img[r*dd:(r+1)*dd, c*dd:(c+1)*dd] = imgI[i].reshape(dd,dd)
   final = img - np.min(img)
   final = (255*(final / np.max(final))).astype(np.uint8)
   np.save('samps/manifold'+str(name)+'.npy', final)
         

if __name__ == '__main__':
   m = 55000
   trainDat, valDat, testDat, D = loadDat()
   trainBatcher = Batcher(trainDat.images[:m], trainDat.images[:m])
   valBatcher = Batcher(valDat.images[:m], valDat.images[:m])
   testBatcher = Batcher(testDat, testDat)

   #Fun with python scoping rules
   #We don't have to directly pass these as args.
   maxIters = 10000000
   batchSize = 100
   valBatchSize = 100
   epoch = 1000
   nil = [D, 2000, 4, 1000, D]
   nz = 1
   auto = 2 #Index in nil
   autoDim = nil[auto]/2
   #Setup TF symbolics
   D = nil[0]
   C = nil[-1]

   with tf.device('/gpu:1'):
      x = tf.placeholder(tf.float32, [None, D])
      W = []
      for l in range(1, len(nil)):
         if l == auto + 1:
            W += [tf.Variable(tf.random_normal(([nil[l-1]/2, nil[l]]), stddev=1e-2))]
         else:
            W += [tf.Variable(tf.random_normal(([nil[l-1], nil[l]]), stddev=1e-2))]
      y = tf.placeholder(tf.float32, [None, C])
   ###End
 
   #Saver
   saver = tf.train.Saver(max_to_keep=2)
   space = np.load('space.npy')

   opt = optimizer(eta=.001)
   sess = tfInit()
   #saver.restore(sess, 'samps/save100000')
   #saveManifold(25, 'final')
   #print 'Train error: ' + str(evalModel(testBatcher))
   train(trainBatcher, valBatcher)



