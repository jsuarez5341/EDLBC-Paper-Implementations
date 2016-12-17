import sys
import numpy as np
from pdb import set_trace as tt
from itertools import chain

def splitWords(txt):
   words = txt.split(' ')
   words = [list('|'+e) for e in words]
   numWords = len(np.unique(words))
   return words, numWords

def wordPieceBuild(txt, vocabSz):
   words, numOrig = splitWords(txt)
   print(numOrig)

   counts = {}
   for i in range(25):
      counts, words = wordPieceStep(words, counts)
      numWords = len(np.unique(list(chain.from_iterable(words))))
      print(numWords)
   counts, words = wordPieceStep(words, counts, vocabSz=vocabSz)
   numWords = len(np.unique(list(chain.from_iterable(words))))
   print(numWords)

   return counts

def wordPieceStep(words, counts, vocabSz=None):
   counts = {**counts, **buildCounts(words, vocabSz)}
   #counts = buildCounts(words, vocabSz)
   newWords = applyCounts(counts, words)
   return counts, newWords

def buildCounts(words, vocabSz=None):
   counts = {}

   for word in words:
      for ind in range(len(word)-1):
         bigram = ''.join(tuple(word[ind:ind+2]))
         if bigram in counts:
            counts[bigram] += 1
         else:
            counts[bigram] = 1

   countsList = sorted([(k, v) for k, v in counts.items()], key = lambda x: x[1], reverse=True)

   numKeep = 0
   if vocabSz is None:
      numKeep = int(0.02* len(countsList))
   else:
      numKeep = vocabSz

   countsList = countsList[:numKeep]
   counts = dict(countsList)

   return counts

def applyCounts(counts, words):
   newWords = []
   for word in words:
      outWord = [] 
      ind = -1
      while ind < len(word):
         ind += 1
         for attempt in np.arange(ind, len(word))[::-1]:
            if attempt == ind:
               outWord += [word[ind]]
               break

            wp = tuple(word[ind:(attempt+1)])
            wp = ''.join(wp)
            if wp in counts:
               outWord += [wp]
               ind = attempt
               break
      newWords += [outWord]

   return newWords       
            
def writeToFile(splitLines, fOut):
   for line in splitLines:
      pass

if __name__ == '__main__':
   fIn = sys.argv[1]
   fOut = open(sys.argv[2], 'w')
   vocabSz = int(sys.argv[3])
   
   txt = open(fIn).read().lower().splitlines()
 
   print('Building')
   counts = wordPieceBuild(' '.join(txt[:50000]), vocabSz)

   print('Applying')
   wordsCounted = []
   ind = 0
   for line in txt:
      ind += 1
      if ind % 10000 == 0:
         print(ind)
      words = ['|'+e for e in line.split(' ')]
      post = applyCounts(counts, line.split(' '))
      post = list(chain.from_iterable(post))
      fOut.write(' '.join(post) + '\n')

