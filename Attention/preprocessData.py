import numpy as np
import sys
from pdb import set_trace as t
import h5py

UNK = 0
START = 1
STOP = 2

def buildIndDict(txt, maxWords=10000):
   #Prints for load time
   print 'Total sentences: ' + str(len(txt))
   nowReadingInd = 0
   indDict = {}
    
   for sent in txt:
      #Prints for load time
      nowReadingInd += 1
      if nowReadingInd % 10000 == 0:
         print 'Now reading sentence: ' + str(nowReadingInd)

      indexedSentence = []
      for word in sent.split(' '):
         try:
            _ = indDict[word]
         except:
            indDict[word] = UNK
         indDict[word] += 1

   #Take top-k most common
   indDict= [(k, indDict[k]) for k in indDict.keys()]
   #Not a dict here, but avoids additional declarations
   indDict.sort(key=lambda x: x[1], reverse=True)
   indDict = dict(indDict[:maxWords-3]) #minus three for start, end, unk

   #Replace frequencies by index
   i = 3
   for k in indDict.keys():
      indDict[k] = i
      i += 1
   indDict['UNK'] = UNK
   indDict['START'] = START
   indDict['STOP'] = STOP

   return indDict

def fixLen(src, dst, l=30):
   outSrc= []
   outDst= []
   for i in range(len(src)):
      srcSent = src[i]
      dstSent = dst[i]
      if len(srcSent) > l or len(dstSent) > l:
         continue
      '''
      srcSent = srcSent[:l-1] + [STOP]
      dstSent = dstSent[:l-1] + [STOP]
      outSrc += [srcSent]
      outDst += [dstSent]
      '''
      #Pad src
      diff = l - len(srcSent)
      outSrc += [src[i]+ diff*[STOP]]
      #Pad Dst
      diff = l - len(dstSent)
      outDst += [dst[i]+ diff*[STOP]]
   return np.asarray(outSrc), np.asarray(outDst)

def loadMonoCorpus(fName, vocab=1000, verbose=True):
   txt = open(fName).read().lower().splitlines()

   print 'Loading corpus of length: ' + str(len(txt))
   print 'Building index dictionary...'
   indDict = buildIndDict(txt, vocab)
   print 'Built'

   indexedCorpus = []
   nowReadingInd = 0

   print 'Encoding corpus by index...'
   for sent in txt:
      indexedSent = []
      nowReadingInd += 1
      if nowReadingInd % 10000 == 0:
         #Prints for load time
         print 'Indexing sentence: ' + str(nowReadingInd)

      for word in sent.split(' '):
         indexedWord = 0
         try:
            indexedWord = indDict[word]
         except:
            pass
         indexedSent += [indexedWord]

      #Add start/stop
      indexedSent = [START] + indexedSent + [STOP]

      indexedCorpus += [indexedSent]
   return np.asarray(indexedCorpus)
  
if __name__ == '__main__':
   fixedLen = True 
   #Thresholds for train/val/test
   trainPercent = 0.8
   valPercent = 0.9
   testPercent = 1.0

   fNameSrc = sys.argv[1]
   fNameDst = sys.argv[2]
   vocabSrc = 50000
   vocabDst = 25000
   srcCorp = loadMonoCorpus(fNameSrc, vocab=vocabSrc)
   dstCorp = loadMonoCorpus(fNameDst, vocab=vocabDst)

   #Save a lookup dict
   txt = open(fNameDst).read().lower().splitlines()

   print 'Building index dictionary...'
   indDict = buildIndDict(txt, vocabDst)
   wordDict = dict((v, k) for k, v in indDict.iteritems())
   saveMe = np.asarray([wordDict])
   np.save('lookupDict.npy', saveMe)

   txt = open(fNameSrc).read().lower().splitlines()
   indDict = buildIndDict(txt, vocabSrc)
   wordDict = dict((v, k) for k, v in indDict.iteritems())
   saveMe = np.asarray([wordDict])
   np.save('lookupDictSrc.npy', saveMe)
 
   if fixedLen:
      srcCorp, dstCorp = fixLen(srcCorp, dstCorp)
   print 'Done. Saving everything to disk'

   #Compute train/val/test splits
   inds = np.arange(len(srcCorp))
   inds = np.random.permutation(inds)
   m = len(inds)
   trainEnd = int(m * trainPercent)
   valEnd = int(m * valPercent)
   testEnd = int(m * testPercent)

   #Split data
   trainSrc = srcCorp[inds[:trainEnd]]
   trainDst = dstCorp[inds[:trainEnd]]

   valSrc = srcCorp[inds[trainEnd:valEnd]]
   valDst = dstCorp[inds[trainEnd:valEnd]]
   
   testSrc = srcCorp[inds[valEnd:testEnd]]
   testDst = dstCorp[inds[valEnd:testEnd]]

   rootdir = 'datFixed'

   np.save(rootdir+'/trainSrc.npy', trainSrc)
   np.save(rootdir+'/valSrc.npy', valSrc)
   np.save(rootdir+'/testSrc.npy', testSrc)

   np.save(rootdir+'/trainDst.npy', trainDst)
   np.save(rootdir+'/valDst.npy', valDst)
   np.save(rootdir+'/testDst.npy', testDst)
