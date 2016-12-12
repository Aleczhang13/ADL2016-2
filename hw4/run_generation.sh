file="./dl_hw4_data.zip"
if [ -f "$file" ]
then
	  echo "$file found."
else
	  echo "$file not found."
    wget -O hw4_data.zip https://www.dropbox.com/s/xu1fei89vckjjlp/dl_hw4_data.zip?dl=0
    unzip hw4_data.zip
fi
python predict_nlg.py $@
