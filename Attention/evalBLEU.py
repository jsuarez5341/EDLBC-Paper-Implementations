from pdb import set_trace as t
import os

def postprocess(fName, outName):
   txt = open(fName).read()
   blacklist = ['UNK', 'START', 'STOP']
   f = open(outName, 'w')
   for sentence in txt.splitlines():
      for word in sentence.split(' '):
         if word not in blacklist:
            f.write(word + ' ')
      f.write('\n')
   f.close()

if __name__ == '__main__':
   refName = 'translatedTargs.txt'
   hypName = 'translatedSentences.txt'

   postProcRef = postprocess(refName, 'refPost.txt')
   postProcHyp = postprocess(hypName, 'hypPost.txt')

   os.system('perl bleu.pl refPost.txt < hypPost.txt')
