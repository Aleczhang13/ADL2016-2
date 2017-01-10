file="dl_final_data.zip"
if [ -f "$file" ]
then
	  echo "$file found."
else
	  echo "$file not found."
    wget -O dl_final_data.zip https://www.dropbox.com/s/ctheqrh8nllhzek/dl_final_squad_data.zip?dl=0
    unzip dl_final_data.zip
fi
pip3.5 install python-levenshtein tqdm --user
python3.5 -m nltk.downloader -d $HOME/nltk_data punkt
python3.5 pre_test.py $1
python3.5 -m squad.prepro
python3.5 -m basic.cli --len_opt --cluster --answer_path data/raw_ans.txt  --model_path model/toefl-model
python3.5 includeWord.py data/raw_ans.txt $1 $2
