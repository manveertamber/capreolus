punc=True
camel=True
key=True

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
			-ans True
			-o csn_final_stat_fix -l $1 -punc=True $camel_str $key_str" 
		echo $cmd
		echo $logfile

		python $cmd > "withans-$logfile"
	done
done
