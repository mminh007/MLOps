import os
from pathlib import Path
from config_args import setup_parse, update_config


def main(args):
	"""
	"""
	SRC_DIR = Path(args.src_dir)
	TMP_DIR = Path(args.tmp_dir)

	DATA_DIR = SRC_DIR / args.data

	cmd_0 = f"unzip {DATA_DIR} -d {TMP_DIR}"
	os.system(cmd_0)

	# train_loader, val_loader, data_test = build_dataset(args)

	# return train_loader, val_loader, data_test


if __name__ == "__main__":
	parser = setup_parse()

	args = parser.parse_args()
	args = update_config(args)

	main(args)


