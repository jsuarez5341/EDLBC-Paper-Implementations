from pdb import set_trace as t
import numpy as np
import sys

START = 0
STOP = 1
UNK = 2

def buildVocab(txt, maxWords=10000):
   maxWords -= 3
   i = 2
   seen = {}
   counts = {}
   for sent in txt.splitlines():
      for word in sent.split(' '): 
         try:
            _ = seen[word]
            counts[word] += 1
         except:
            seen[word] = 1
            counts[word] = 1
            i += 1

   countAry = [e for e in counts.items()]
   countAry.sort(reverse=True, key = lambda x: x[1])
   countAry = countAry[:maxWords]

   wordDict = {}
   wordDict['STARTTOKEN'] = START
   wordDict['STOPTOKEN'] = STOP
   wordDict['UNKTOKEN'] = UNK
   i = 3
   for e in countAry:
      wordDict[e[0]] = i
      i += 1

   indDict = dict((v, k) for k, v in wordDict.iteritems())
   return wordDict, indDict

def lenMask(indexedSrc, indexedDst, maxLen=30):
   srcMask = [len(e)<(maxLen) for e in indexedSrc]   
   dstMask = [len(e)<(maxLen-1) for e in indexedDst]   
   src = []
   dst = []
   for i in range(len(srcMask)):
      andBool = srcMask[i] * dstMask[i]
      if andBool:
         src += [indexedSrc[i] + [STOP]*(maxLen-len(indexedSrc[i]))]
         dst += [indexedDst[i] + [STOP]*(maxLen-len(indexedDst[i]))]
         #src += [srcMask[i]]
         #dst += [dstMask[i]]
   
   return src, dst

def index(txt, vocab, isSrc=True):
   indexed = []
   for sent in txt.splitlines():
      sentSplit = sent.split(' ')
 
      indexedSent = []
      for word in sentSplit:
         try:
            indexedSent += [vocab[word]]
         except:
            indexedSent += [UNK] 
      indexed += [indexedSent] 
   return indexed

#For verifying correctness
def decode(sent, lookup):
   ret = []
   for e in sent:
      ret += [lookup[e]]
   return ' '.join(ret)

if __name__ == '__main__':
   fSrc = open(sys.argv[1])
   fDst = open(sys.argv[2])
   LANGTOKEN = sys.argv[3]

   rootDir = sys.argv[3]

   maxLenSrc = int(sys.argv[4]) 
   maxLenDst = int(sys.argv[5]) 

   src = fSrc.read().lower()
   dst = fDst.read().lower()

   vocabSrc, lookupSrc = buildVocab(src, maxWords=50000)
   vocabDst, lookupDst = buildVocab(dst, maxWords=25000)

   indexedSrc = index(src, vocabSrc, True)
   indexedDst = index(dst, vocabDst, False)

   indexedSrc, indexedDst = lenMask(indexedSrc, indexedDst, maxLen=30)
   #indexedSrc = [e[::-1] for e in indexedSrc]

   np.save(rootDir + '/src', np.asarray(indexedSrc))
   np.save(rootDir + '/dst', np.asarray(indexedDst))
   np.save(rootDir + '/vocabSrc', lookupSrc)
   np.save(rootDir + '/vocabDst', lookupDst)

   print(decode(indexedSrc[-1], lookupSrc))
   print(decode(indexedDst[-1], lookupDst))

