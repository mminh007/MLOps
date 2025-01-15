import mlflow


def registered_model(client,
                   registered_name: str,
                   model_alias: str,
                   run_id: str):
    try:
        client.create_registered_model(name=registered_name)
        client.get_registered_model(name=model_alias)
    
    except:
        print(f"Model: {registered_name} already exists")
    
    print(f"Create model version: {model_alias}")
    model_uri = f"runs:/{run_id}/pytorch-model"
    mv = client.create_model_version(registered_name, model_uri, run_id)

    # Override 'alias' to the best model version
    print(f"Creating model alias: {model_alias}")
    client.set_registered_model_alias(name=registered_name,
                                        alias=model_alias,
                                        version=mv.version)
    
    print("--Model Version--")
    print("Name: {}".format(mv.name))
    print("Version: {}".format(mv.version))
    print("Aliases: {}".format(mv.aliases))

    

def connect_mlflow(args):
    MLFLOW_TRACKING_URI = args.tracking_uri
    MLFLOW_EXPERIMENT_NAME = args.experiment_name
    
    try:
        mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        print(f"MLFLOW TRACKING URI: {MLFLOW_TRACKING_URI}")
        print(f"MLFLOW EXPERIMENT NAME: {MLFLOW_EXPERIMENT_NAME}")
        
    except Exception as e:
        print(f"Error: {e}")
        
        raise e
        
    