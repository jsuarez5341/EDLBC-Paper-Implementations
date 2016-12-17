import sys

def rewrite(txt, fOut, token):
   fOut = open(fOut, 'w')
   for line in txt:
      fOut.write(token + ' ' + line + '\n')

if __name__ == '__main__':
   fIn = open(sys.argv[1]).read().splitlines()
   fOut = sys.argv[2]
   token = sys.argv[3]

   rewrite(fIn, fOut, token)
   
   
