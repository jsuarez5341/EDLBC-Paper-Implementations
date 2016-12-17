#python wordpiece.py data/german/train.en wpData/german/train.en 25000
#python wordpiece.py data/german/train.de wpData/german/train.de 25000

#python wordpiece.py data/french/train.en wpData/french/train.en 25000
#python wordpiece.py data/french/train.fr wpData/french/train.fr 25000

python appendLang.py data/german/train.en wpData/german/trainMulti.en GERMANTOKEN
python appendLang.py data/french/train.en wpData/french/trainMulti.en FRENCHTOKEN

rm wpData/trainX
rm wpData/trainY

cat data/german/train.de >> wpData/trainX
cat wpData/german/trainMulti.en >> wpData/trainY

cat data/french/train.fr >> wpData/trainX
cat wpData/french/trainMulti.en >> wpData/trainY

echo 'Preprocessing multilingual data'

python2 preprocess.py wpData/trainX wpData/trainY autoWPData 30 30

