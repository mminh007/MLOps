import yaml
import argparse

def setup_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config-file", type=str, help="path to config file")
    parser.add_argument("--data", type=str)
    parser.add_argument("--tmp-dir", type=str)
    parser.add_argument("--src-dir", type=str, help="using save model")
    parser.add_argument("--train-file", type=str)
    parser.add_argument("--val-file", type=str)

    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--imgsz", type=tuple, help="size of input")
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--val-size", type=float)
    
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--version", type=str)
    parser.add_argument("--tracking-uri", type=str)
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--registered-name", type=str)
    parser.add_argument("--model-alias", type=str)

    return parser


def update_config(args: argparse.Namespace):
	if not args.config_file:
		return args

	cfg_path = args.config_file + ".yaml" if not args.config_file.endswith(".yaml") else args.config_file

	with open(cfg_path, "r") as f:
		data = yaml.load(f, Loader=yaml.FullLoader)
    
	for key, value in data.items():
		if getattr(args, key) is None:
			setattr(args, key, value)

    # config_args = argparse.Namespace(**data)
    # args = parser.parse_args(namespace=config_args)
    
	return args