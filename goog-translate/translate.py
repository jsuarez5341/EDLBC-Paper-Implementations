import tensorflow as tf
import numpy as np
import codecs
import time

import batcher
import evalBLEU
from pdb import set_trace as tt

#Tensorflow setup
def tfInit():
   init = tf.initialize_all_variables()
   sess = tf.Session()
   sess.run(init)
   return sess

def loadDat(rootDir='data'):
   #Training set
   X = np.load(rootDir + '/src.npy')
   Y = np.load(rootDir + '/dst.npy')

   #Randomly shuffle data
   inds = np.arange(X.shape[0])
   permInds = np.random.permutation(inds)
   X = X[permInds]
   Y = Y[permInds]


   #XTest = np.load(rootDir + '/srcTest.npy')[:m]
   #YTest = np.load(rootDir + '/dstTest.npy')[:m]

   M = X.shape[0]
   XTrain, YTrain = X[:(.8*M)][:m], Y[:(.8*M)][:m]
   XTest, YTest = X[(.8*M):(.9*M)][:m], Y[(.8*M):(.9*M)][:m]
   XVal, YVal = X[(.9*M):][:m], Y[(.9*M):][:m]

   batchTrain = batcher.Batcher(XTrain, YTrain)
   batchVal = batcher.Batcher(XVal, YVal)
   batchTest = batcher.Batcher(XTest, YTest)

   #Vocabulary
   vocabSrc = np.load(rootDir + '/vocabSrc.npy', encoding='latin1').tolist()
   vocabDst = np.load(rootDir + '/vocabDst.npy', encoding='latin1').tolist()

   vocabSrcSz = len(vocabSrc.keys())
   vocabDstSz = len(vocabDst.keys())

   return batchTrain, batchVal, batchTest, vocabSrc, vocabDst, vocabSrcSz, vocabDstSz

###ATTENTION###
def score(hDst, hSrc):
   return tf.reduce_sum(hDst * hSrc, 1)
   return tf.dot(hDst, hSrc)

def align(hSrcAry, ht):
   denom = sum([tf.exp(score(ht, hs)) for hs in hSrcAry])
   out = [tf.exp(score(ht, hs)) / denom for hs in hSrcAry]
   out = tf.pack(out)
   return out

def attention(hSrcAry, ht):
   alph = align(hSrcAry, ht)
   context = tf.expand_dims(alph, 2) * tf.pack(hSrcAry) 
   return tf.reduce_sum(context, axis=0)
 
def attentionSum(hSrcAry, ht): 
   return tf.reduce_sum(tf.pack(hSrcAry), axis=0)

def attentionQuadForm(hSrcAry, ht):
   hSrc = tf.concat(0, hSrcAry)
   hSrcAry = tf.pack(hSrcAry)
   hSrc = tf.matmul(hSrc, attnW)
   hSrc = tf.reshape(hSrc, (batchSize, T, H))

   htExp   = tf.expand_dims(ht, axis=1)
   scored  = tf.concat(0, tf.unpack(hSrc * htExp))
   scored  = tf.transpose(scored, (1,0))

   scored = tf.reshape(scored, (batchSize, T, H))
   scored = tf.transpose(scored, (0,2,1))
   ht     = tf.expand_dims(ht, axis=1)

   align = tf.batch_matmul(ht, scored)
   align = tf.transpose(align, (2, 0, 1))

   context = tf.reduce_sum(hSrcAry * align, axis=0)
   return context

def attentionGeneral(hSrcAry, ht):
   ht = tf.expand_dims(ht, axis=0)
   scored = tf.reduce_sum(hSrcAry * ht, axis=2)
   aligned = scored
   aligned = tf.expand_dims(aligned, axis=2)
   context = tf.reduce_sum(hSrcAry * aligned, axis=0)
   return context

   '''
   hSrcAry = tf.pack(hSrcAry)
   ht = tf.expand_dims(ht, axis=0)
   scored = tf.reduce_sum(hSrcAry * ht, axis=2)
   #denom = tf.reduce_sum(scored, axis=0)
   aligned = scored #/ denom
   aligned = tf.expand_dims(aligned, axis=2)
   context = tf.reduce_sum(hSrcAry * aligned, axis=0)
   return context
   '''



def attentionComb(hSrcAry, ht):
   #hSrc = tf.pack(hSrcAry)
   #align = tf.batch_matmul(hSrc, attnW)
   #align = tf.reduce_sum(align, 0)
   #return ht * align
   hSrc = tf.pack(hSrcAry)
   hSrc = tf.reshape(hSrc, (T*batchSize, H))
   align = tf.matmul(hSrc, attnW)
   align = tf.reshape(align, (T, batchSize, H))
   align = tf.reduce_sum(align, 0)
   return ht * align
 
#################

def CESoftmax(a, y, off=1e-4):
   a = tf.nn.softmax(a)
   a = tf.clip_by_value(a, off, 1-off)
   ce = -(y*tf.log(a) + (1-y)*tf.log(1-a))
   return ce

def loss():
   lbl = tf.one_hot(y, vocabDstSz)
   a=encDecTrain
   a = tf.transpose(tf.pack(a), (1,0,2))
   
   mask = tf.sequence_mask(lengths, maxlen=T)
   mask = tf.to_float(tf.expand_dims(mask, 2))

   mask = tf.concat(0, tf.unpack(mask))
   lbl = mask * tf.concat(0, tf.unpack(lbl))
   a = mask * tf.concat(0, tf.unpack(a))

   cost = tf.nn.softmax_cross_entropy_with_logits(a, lbl)   
   #costPerExample = tf.reduce_sum(tf.reshape(cost, shape=[T, batchSize]), 0)
   
   return tf.reduce_sum(cost) / tf.to_float(tf.reduce_sum(lengths))

def optimizer(eta):
   agg = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
   return tf.train.AdamOptimizer(eta).minimize(loss(),
         aggregation_method=agg)

def encodeDecodeTrain():
   return decodeTrain(*encode())

def encodeDecodeTest():
   return decodeTest(*encode())

def encode():
   embeddings = tf.nn.embedding_lookup(embedSrc, x) 
   embeddings = tf.unpack(embeddings, axis=1) 
   s = sInit
   oList = []
   for t in range(T):
      if t > 0:  
         tf.get_variable_scope().reuse_variables()

      inp = embeddings[t]
      o, s = deepEncodeLSTM(inp, s)
      oList += [o]

   return oList, s

def decodeTrain(oEncodeList, s):
   oList = []
   inp = tf.nn.embedding_lookup(embedDst, [0]*batchSize)
   for t in range(T):
      if t > 0: 
         tf.get_variable_scope().reuse_variables()

      o, s = deepDecodeLSTM(inp, s)

      c = attentionGeneral(oEncodeList, o)
      oc = tf.concat(1, [o,c])

      oProj = tf.matmul(oc, W) + b

      inp = c + tf.nn.embedding_lookup(embedDst, y[:, t])
      
      oList += [oProj]

   return oList

def decodeTest(oEncodeList, s):
   oList = []
   inp = tf.nn.embedding_lookup(embedDst, [0]*batchSize)
   #inp = tf.concat(1, [inp, tf.zeros((batchSize, H))])
   for t in range(T):
      if t > 0: 
         tf.get_variable_scope().reuse_variables()

      o, s = deepDecodeLSTM(inp, s)

      c = attentionGeneral(oEncodeList, o)
      oc = tf.concat(1, [o,c])

      oProj = tf.matmul(oc, W) + b
   
      pred = tf.argmax(tf.nn.softmax((oProj)), 1)
      inp = c + tf.nn.embedding_lookup(embedDst, pred)
      oList += [pred]

   return oList


def train(trainBatcher, valBatcher):
   npTrain     = []
   npVal       = []
   npTrainBleu = []
   npValBleu   = []
   for i in range(maxIters):
      xTrain, yTrain, trainLengths = trainBatcher.next(batchSize)
      xVal, yVal, valLengths       = valBatcher.next(batchSize)

      if i % 25 == 0:
         trainCost = sess.run(loss(), feed_dict={x:xTrain, y:yTrain, lengths: trainLengths})
         valCost   = sess.run(loss(), feed_dict={x:xVal, y:yVal, lengths: valLengths})

         bleuTrain = testModel(trainBatcher) #Switch to test
         bleuVal   = testModel(valBatcher) #Switch to test

         print(str(i) + '| Training Error: ' + str(trainCost)) 
         print(str(i) + '| Validation Error: ' + str(valCost))
         print(str(i) + '| Training Bleu: ' + str(bleuTrain))
         print(str(i) + '| Validation Bleu: ' + str(bleuVal))
         
         npTrain     += [trainCost]
         npVal       += [valCost]
         npTrainBleu += [bleuTrain]
         npValBleu   += [bleuVal]

      sess.run(opt, feed_dict={x:xTrain, y:yTrain, lengths: trainLengths})

      if i % 250 == 0 and i != 0:
         print('Saving checkpoint')
         saver.save(sess, 'savedModels/model.ckpt', global_step=i)
         np.save('savedModels/npTrain.npy', npTrain)
         np.save('savedModels/npVal.npy', npVal)
         np.save('savedModels/npTrainBleu.npy', npTrainBleu)
         np.save('savedModels/npValBleu.npy', npValBleu)
         print('Done saving')

def testModel(testBatcher):
   outPreds= codecs.open('translatedSentences.txt', 'w', 'utf-8')
   outTargs = codecs.open('translatedTargs.txt', 'w', 'utf-8')
   outRef= codecs.open('translatedRef.txt', 'w', 'utf-8')

   for i in range(testIters):
      xTest, yTest, lengths = testBatcher.next(batchSize, movePointer=True)
      #Hack to enable test mode.
      transMat = sess.run(encDecTest, feed_dict={x:xTest, y:yTest})
      transMat = np.asarray(transMat).T

      #One sentence per row 
      R, C = transMat.shape
      for r in range(R):
         for c in range(lengths[r]):
            wordIndPred = transMat[r,c]
            wordIndTarg= yTest[r,c]
            wordIndRef= xTest[r,c]
            wordPred = vocabDst[wordIndPred]
            wordTarg = vocabDst[wordIndTarg]
            wordRef= vocabSrc[wordIndRef]
            outPreds.write(wordPred + ' ')
            outTargs.write(wordTarg + ' ')
            outRef.write(wordRef + ' ')
         outPreds.write('\n')
         outTargs.write('\n')
         outRef.write('\n')

   outPreds.close()
   outTargs.close()
   outRef.close()

   evalBLEU.evalBleu() 
   bleu = open('bleuScore.txt') 
   bleuScore = float(bleu.read())
   bleu.close()
   return bleuScore
      
if __name__ == '__main__':
   testMode = False
   restore = False
   restoreIter = 4000
   testIters = 100

   m = 1000 #Max per train/val/test
   T = 30
   H = 256
   batchSize = 64
   maxIters = 100010
   numLayers = 2
   embedDim = 256
   eta = 0.001
   savename = 'NMTeng2viet'

   x = tf.placeholder(tf.int32, [batchSize, T])
   y = tf.placeholder(tf.int32, [batchSize, T])
   lengths = tf.placeholder(tf.int32, [batchSize])
   # [tf.constant(np.zeros((batchSize, H), np.float32)) for i in range(numLayers)]
 
   trainBatcher, valBatcher, testBatcher, vocabSrc, vocabDst, vocabSrcSz, vocabDstSz = loadDat('autoWPData')

   embedSrc = tf.Variable(tf.random_normal((vocabSrcSz, embedDim), stddev=1e-2))
   embedDst = tf.Variable(tf.random_normal((vocabDstSz, embedDim), stddev=1e-2))

   encodeLSTM = tf.nn.rnn_cell.BasicLSTMCell(H)
   deepEncodeLSTM = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.MultiRNNCell([encodeLSTM] * numLayers), input_keep_prob=.8)

   decodeLSTM = tf.nn.rnn_cell.BasicLSTMCell(H)
   deepDecodeLSTM = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.MultiRNNCell([decodeLSTM] * numLayers), input_keep_prob=.8)
   sInit = deepEncodeLSTM.zero_state(batchSize, tf.float32)

   #Projection params
   W = tf.Variable(tf.random_normal((2*H, vocabDstSz), stddev=1e-2))
   b = tf.Variable(tf.random_normal((batchSize, 1), stddev=1e-2))
   attnW = tf.Variable(tf.random_normal((H, H), stddev=1e-2))
   h = tf.Variable(tf.random_normal((T,1,1), stddev=1e-2))

   encDecTrain = encodeDecodeTrain()
   encDecTest  = encodeDecodeTest()
   if not testMode:
      opt = optimizer(eta)   

   print('Initializing Saver')
   saver = tf.train.Saver(max_to_keep=2)

   sess = tfInit()
   if restore:
      saver.restore(sess, 'savedModels/model.ckpt-' + str(restoreIter))

   if not testMode:
      train(trainBatcher, valBatcher)
   else:
      print(testModel(trainBatcher))
      print(testModel(valBatcher))
      print(testModel(testBatcher))

   

'''
def decodeTrain(oEncodeList, s):
   tt()
   oList = []
   inp = tf.nn.embedding_lookup(embedDst, [0]*batchSize)
   for t in range(T):
      if t > 0: 
         tf.get_variable_scope().reuse_variables()
         if not testMode:
            inp = tf.nn.embedding_lookup(embedDst, y[:, t-1])

      o, s = deepDecodeLSTM(inp, s)

      #c = attention(oEncodeList, o)
      #oc = tf.concat(1, [o,c])

      oProj = tf.matmul(o, W) + b
      
      if not testMode:
         oList += [oProj]
      else: 
         pred = tf.argmax(tf.nn.softmax((oProj)), 1)
         oList += [pred]
         inp = tf.nn.embedding_lookup(embedDst, pred)

   return oList

'''
