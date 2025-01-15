from config_args import setup_parse, update_config
from src.data import build_dataset
from src.modeling import ResNet50, evaluate
from src.ultis import connect_mlflow
import os
import torch
import torch.nn as nn
import mlflow
from mlflow.pytorch import log_model
from pathlib import Path
import numpy as np
import time


def main(args, **kwargs):
	
	connect_mlflow(args)

	SRC_DIR = Path(args.src_dir)
	TMP_DIR = Path(args.tmp_dir)

	train_loader, val_loader, data_test = build_dataset(args)

	model = ResNet50(num_classes= args.num_classes)
	model.to(args.device)

	criterion = nn.CrossEntropyLoss()
	
	if args.optimizer == "Adam":
		optimizer = torch.optim.Adam(model.parameters(),
							   lr=args.lr,
							   weight_decay=args.weight_decay)
		
	elif args.optimizer == "AdamW":
		optimizer = torch.optim.AdamW(model.parameters(),
								lr = args.lr,
								weight_decay=args.weight_decay)
		
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
											 step_size=args.epochs * 0.5,
											 gamma=0.1)
	
	with mlflow.start_run(run_name=args.run_name) as run:
		print(f"MLFLOW run_id: {run.info.run_id}")
		print(f"MLFLOW experiment_id: {run.info.experiment_id}")
		print(f"MLFLOW run_name: {run.info.run_name}")

		mlflow.set_tag(
			{
				"Model's version": args.version
			}
		)

		mlflow.log_params(
			{
				"input_size": args.imgsz,
				"batch_size": args.batch_size,
				"epochs": args.epochs,
				"lr": args.lr,
				"optimizer": args.optimizer
			}
		)

		best_model_info = {
			"model_state_dict": None,
			"optimizer_state_dict": None,
			"best_loss": None,
			"epoch": None,
		}

		best_loss = np.inf
		for epoch in range(args.epochs):
			start = time.time()
			batch_losses = []

			model.train()
			for image, label in train_loader:
				x = image.to(args.device)
				y = label.long().to(args.device)

				optimizer.zero_grad()
				output = model(x)

				loss = criterion(output, y)
				loss.backward()

				optimizer.step()
				batch_losses.append(loss.item())

			epoch_loss = sum(batch_losses) / len(train_loader)
			mlflow.log_metric("training_loss", f"{epoch_loss: .6f}", step=epoch)

			val_loss = evaluate(model, val_loader, args.device, criterion)
			mlflow.log_metric("val_loss", f"{val_loss: .6f}", step=epoch)

			if val_loss < best_loss:
				best_loss = val_loss
				best_model_info.update(
					{
						"model_state_dict": model.state_dict(),
						"optimizer_state_dict": optimizer.state_dict(),
						"best_loss": val_loss,
						"epoch": epoch
					}
				)
			print(f"EPOCH {epoch + 1}: \tTrain loss: {epoch_loss: .4f} \tVal loss: {val_loss: .4f}")

			end = time.time()
			scheduler.step()
			print(f"Time taken: {end - start}")

		torch.save(best_model_info, TMP_DIR / "model.pth")

		local_artifacts = SRC_DIR / run.info.run_id
		local_artifacts.mkdir(parents=True, exists_ok=True)

		cmd_0 = f"cp {TMP_DIR} / model.pth {SRC_DIR}/model.pth"
		os.system(cmd_0)


		model.load_state_dict(best_model_info["model_state_dict"])
		test_loss = evaluate(model, data_test, args.device, criterion)
		model.log_metric("Test_loss", f"{test_loss: .6f}")
		print(f"Test loss: {test_loss}")

		# log model to mlflow sever
		log_model(model, artifact_path="RestNet50",
				pip_requirements= "./requirements.txt")
		
		mlflow.log_artifact("./configs/parameters.yaml", artifact_path="config")

		# using xcom_push to pass data between tasks.
		kwargs["ti"].xcom_push(key="run_id", value = run.info.run_id)
		kwargs["ti"].xcom_push(key="test_loss", value = test_loss)

		print("Training Completed!")
		mlflow.end_run()


if __name__ == "__main__":
	
	parser = setup_parse()

	args = parser.parse_args()
	args = update_config(args)

	main(args)


