import os
from pathlib import Path
from config_args import setup_parse, update_config


def main(args):
	"""
	"""
	SRC_DIR = Path(args.src_dir)
	TMP_DIR = Path(args.tmp_dir)

	DATA_DIR = SRC_DIR / args.data
	
	csv_file = TMP_DIR / "train.csv"

	#os.system(f"sudo chown -R $(whoami):$(whoami) {DATA_DIR}") 
	cmd_1 = f"chown -R $(whoami):$(whoami) ./opt/airflow/DATA/{args.data}" #"chmod u+w ~/cls_temp_dir"
	#cmd_2 = "ls -l"
	cmd_3 = f"unzip {DATA_DIR} -d {TMP_DIR}"
	#os.system(cmd_0)

	os.system(cmd_1)

	if csv_file.exists():
		print(f"File {csv_file} is existed!!!")
	
	else:		
		os.system(cmd_3)
	


if __name__ == "__main__":
	parser = setup_parse()

	args = parser.parse_args()
	args = update_config(args)

	main(args)


