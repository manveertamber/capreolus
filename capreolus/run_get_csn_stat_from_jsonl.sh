punc=True
camel=True
key=True

lang=$1
output_dir=$2

for camel in False True 
do
	if [ $camel = True ]
	then
		logprefix="camel"
		camel_str="-camel=True"
	else
		logprefix="nocamel"
		camel_str=""
	fi

	for key in False True 
	do
		if [ $key = True ]
		then
			logfile="$logprefix-withkey.log"
			key_str="-key=True"
		else
			logfile="$logprefix-nokey.log"
			key_str=""
		fi

		cmd="contrib/get_csn_stat_from_jsonl.py 
			-o $output_dir -l $lang -punc=True $camel_str $key_str" 
		echo $cmd
		echo $logfile

		python $cmd > $logfile
	done
done
