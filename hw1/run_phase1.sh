python glove.py $1
python filterVocab.py fullVocab.txt <raw_glove> $2filter_glove.txt
python w2v.py $1
python filterVocab.py fullVocab.txt <raw_w2v> $2filter_word2vec.txt
