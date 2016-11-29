import tensorflow as tf 
import numpy as np 
from toyDataGen import genDat
import batcher
from pdb import set_trace as tt

#Tensorflow setup
def tfInit():
   init = tf.initialize_all_variables()
   sess = tf.Session()
   sess.run(init)
   return sess

#Sums loss over a set of predictions
def loss():
   #Make some damn hot one hots
   predLogits = encodeDecode()
   lbl = tf.cast(tf.one_hot(tf.cast(y, tf.int64), vocabOut, 1, 0), tf.float32)

   #meanLoss = CELoss(predLogits, lbl)
   #return meanLoss
   #Gotcha: softmax cross entropy expects 2d tensors
   predLogits = tf.concat(0, tf.unpack(predLogits))
   lbl = tf.concat(0, tf.unpack(lbl))
   lossAry = tf.nn.softmax_cross_entropy_with_logits(predLogits, lbl)
   return tf.reduce_mean(lossAry)

def CELoss(a, y, off=1e-4):
   a = tf.clip_by_value(a, off, 1-off)
   ce = y*tf.log(a) + (1-y)*tf.log(1-a)
   ceSum = -tf.reduce_sum(ce, (1, 2))
   return tf.reduce_mean(ceSum)

def score(hDst, hSrc):
   return tf.reduce_sum(hDst * hSrc, 0)
   return tf.dot(hDst, hSrc)

def align(hSrcAry, hDst):
   denom = 0
   for e in hSrcAry:
      denom += tf.exp(score(hDst, e))
   
   out = []
   for e in hSrcAry:
      out += [tf.exp(score(hDst, e)) / denom]
   out = tf.transpose(tf.pack(out))
   return out

def getContext(hSrcAry, alph):
   ciVec = 0
   N = len(hSrcAry)
   for i in range(N):
      ciVec += (1.0/N) * alph[:, i] * hSrcAry[i]
   return ciVec
   
   
def encodeDecode():
   sNew = s
   err = 0.0
   hSAry = []
   for t in range(T): 
      if t > 0:
         tf.get_variable_scope().reuse_variables()

      #LSTM Timestep activation
      inp = tf.nn.embedding_lookup(embeddingsEncode, x[:, t])
      o, sNew = deeplstmEncode(inp, sNew)
      #o, sNew = deeplstm(x[:,t:t+1], sNew)
      hSAry += [o]

   lenDst = y._shape_as_list()[1] 
   preds = []
   allInds = []
   
   inp = tf.nn.embedding_lookup(embeddingsDecode, batchSize*[START])
   for t in range(T):
      #LSTM Timestep activation
      o, sNew = deeplstmDecode(inp, sNew)
 
      #o = hd. Implement attention
      alph = align(hSAry, o)
      c = getContext(hSAry, alph)
      oc = tf.concat(1, [o,c])
      #oc = o
   
      #Compute predictions and loss
      #logits = tf.matmul(o, W) + b
      logits = tf.matmul(oc, W) + b
      pred = logits#tf.nn.softmax(logits)
      preds += [logits]

      if not test:
         inp = tf.nn.embedding_lookup(embeddingsDecode, y[:, t])
      else:
         #Use pred to encode next layer input
         inds = tf.argmax(pred, 1)
         allInds += [inds]
         inp = tf.nn.embedding_lookup(embeddingsDecode, inds)

   #Some save code for test mode
   if test:
       translations = tf.transpose(tf.pack(allInds))
       return translations

   predLogits = tf.transpose(tf.pack(preds), perm=[1,0,2])
   return predLogits

def optimizer(err):
   return tf.train.AdamOptimizer(eta).minimize(err, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
   return tf.train.GradientDescentOptimizer(eta).minimize(err)


def train(trainGen, valGen, batchSize, maxIters=100):
   saveTrainErrs = []
   saveValErrs = []
   meanValBest = 9999999
   sumCost = 0
   for i in range(maxIters):
      xBatch, yBatch = trainGen.next(batchSize)
      #sNp = np.zeros((batchSize, deeplstmEncode.state_size), np.float32)

      _, cost = sess.run([opt, err], feed_dict={eta: 0.001, x:xBatch, y:yBatch})

      xBatch, yBatch = valGen.next(batchSize)
      #sNp = np.zeros((batchSize, deeplstmEncode.state_size), np.float32)
      costVal = sess.run(err, feed_dict={x:xBatch, y:yBatch})
 
      saveTrainErrs += [cost]
      saveValErrs += [costVal]

      if i % 10 == 0:
         print str(i) + ', train: ' + str(cost) + ', val: ' + str(costVal)
      if i % 100 == 0:
         print 'Saving error'
         logRoot = 'tensorboardLogs/'
         np.save(logRoot + 'trainErr.npy', saveTrainErrs)
         np.save(logRoot + 'valErr.npy', saveValErrs)
      
      if i != 0 and i % 200 == 0:
         print 'Saving run'
         saver.save(sess, 'runsaves/'+savename, global_step=i)

def testModel(gen, batchSize, maxIters=100):
   for i in range(maxIters):
      xBatch, yBatch = gen.next(batchSize)
      transMat = sess.run(transPredictor, feed_dict={x:xBatch, y:yBatch})
      #One sentence per row
      R, C = transMat.shape
      for r in range(R):
         for c in range(C):
            wordIndPred = transMat[r,c]
            wordIndTarg= yBatch[r,c]
            wordIndRef= xBatch[r,c]
            wordPred = vocabDict[wordIndPred]
            wordTarg = vocabDict[wordIndTarg]
            wordRef= vocabDictSrc[wordIndRef]
            outPreds.write(wordPred + ' ')
            outTargs.write(wordTarg + ' ')
            outRef.write(wordRef + ' ')
         outPreds.write('\n')
         outTargs.write('\n')
         outRef.write('\n')


if __name__ == '__main__':
   test = False
   resto = True
   resoIter = 16400
   if test: 
      outPreds= open('translatedSentences.txt', 'w')
      outTargs = open('translatedTargs.txt', 'w')
      outRef= open('translatedRef.txt', 'w')

   m = 130000
   T = 30 #Max vector length, padded
   H = 256
   batchSize = 16
   maxIters = 100010
   numLayers = 4
   embedDim = 128
   vocabIn = 47861
   vocabOut = 22731
   savename = 'NMTeng2viet'
   #Load in data
   X = np.load('datFixed/testSrc.npy')[:m]
   Y = np.load('datFixed/testDst.npy')[:m]
   batchTrain = batcher.Batcher(X, Y)

   #Validation set
   X = np.load('datFixed/valSrc.npy')[:m]
   Y = np.load('datFixed/valDst.npy')[:m]
   batchVal = batcher.Batcher(X, Y)
   #Test vars
   vocabDict = np.load('lookupDict.npy')[0]
   vocabDictSrc = np.load('lookupDictSrc.npy')[0]

   print 'Initializing Variables'
   #Initialize variables
   UNK = 0
   START = 1
   STOP = 2

   lstmEncode = tf.nn.rnn_cell.BasicLSTMCell(H)
   deeplstmEncode = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.MultiRNNCell([lstmEncode] * numLayers), input_keep_prob=0.8)
   #deeplstmEncode = tf.nn.rnn_cell.MultiRNNCell([lstmEncode] * numLayers)

   lstmDecode = tf.nn.rnn_cell.BasicLSTMCell(H)
   deeplstmDecode = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.MultiRNNCell([lstmDecode] * numLayers), input_keep_prob=0.8)
   #deeplstmDecode = tf.nn.rnn_cell.MultiRNNCell([lstmDecode] * numLayers) 

   #W = tf.Variable(tf.random_normal((H, vocabOut), stddev=1e-2))
   W = tf.Variable(tf.random_normal((2*H, vocabOut), stddev=1e-2))
   b = tf.Variable(tf.random_normal((batchSize, 1), stddev=1e-2))
   o = tf.zeros([batchSize, H])

   x = tf.placeholder(tf.int32, [batchSize, T])
   y = tf.placeholder(tf.int32, [batchSize, T])
   eta  = tf.placeholder(tf.float32, shape=[])
   #s = tf.placeholder(tf.float32, [batchSize, deeplstmEncode.state_size])
   s = tf.constant(np.zeros((batchSize, deeplstmEncode.state_size), np.float32))

   #embEncode = np.load('dataEncode/embeddings60000.npy')
   #embeddingsEncode = tf.Variable(embEncode)
   embeddingsEncode = tf.Variable(tf.random_normal((vocabIn, embedDim), stddev=1e-2))

   #embDecode = np.load('dataDecode/embeddings300000.npy')
   #embeddingsDecode = tf.Variable(embDecode)
   embeddingsDecode = tf.Variable(tf.random_normal((vocabOut, embedDim), stddev=1e-2))

   #Variable initialization
   if not test:
      print 'Initializing Graphs'
      err = loss()
      print 'Initializing Optimizer'
      opt = optimizer(err)
   else:
      transPredictor = encodeDecode()

   print 'Initializing Saver'
   saver = tf.train.Saver(max_to_keep=2)

   sess = tfInit()

   #print 'Loading previous run variables'
   if resto:
      saver.restore(sess, 'runsaves/'+savename+'-'+str(resoIter))

   if not test:
      print 'Training'
      train(batchTrain, batchVal, batchSize, maxIters)
   else:
      print 'Testing'
      testModel(batchTrain, batchSize, maxIters)

   '''
   xTest, yTest = batchTrain.next(batchSize)
   sNp = np.zeros((batchSize, deeplstmEncode.state_size), np.float32)
   print xTest
   print yTest
   print sess.run(encodeDecode(), feed_dict={x:xTest, y:yTest, s:sNp})
   '''

   






