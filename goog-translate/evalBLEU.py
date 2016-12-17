from pdb import set_trace as t
import os

def postprocess(fName, outName):
   txt = open(fName, encoding='utf-8').read()
   blacklist = ['STARTTOKEN', 'STOPTOKEN']
   f = open(outName, 'w', encoding='utf-8')
   for sentence in txt.splitlines():
      for word in sentence.split(' '):
         if word not in blacklist:
            f.write(word + ' ')
      f.write('\n')
   f.close()

def evalBleu():
   refName = 'translatedTargs.txt'
   hypName = 'translatedSentences.txt'

   postProcRef = postprocess(refName, 'refPost.txt')
   postProcHyp = postprocess(hypName, 'hypPost.txt')

   os.system('perl bleu.pl refPost.txt < hypPost.txt')

if __name__ == '__main__':
   evalBleu()

