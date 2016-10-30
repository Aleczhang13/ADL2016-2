hw1
===============
##environment
python3.5.0

Ubuntu 14.04

##word2vec

檔案寫在 `w2v.py` 裡面

參考了 tensorflow 官方的[教學]('https://www.tensorflow.org/versions/r0.11/tutorials/word2vec/index.html')
使用底下提供的 optimize 版本，並把一些用不到的 method 拿掉


##gloves

檔案是 `glove.py`

參考了這個 [Tensorflow-glove]('https://github.com/GradySimon/tensorflow-glove')
也是修改了一些部分

## ptt
使用 `w2v.py` 

但有先對 `ptt_corpus` 做前處理

由於 `w2v.py` 可以篩選每個字出現的最少次數，所以對於出現在 `ptt_corpus` 裡面的 `stopwords`，把他改成一個隨機不重複的整數，然後設定 `min_count =  2`，就可以把 stopwords 濾掉


