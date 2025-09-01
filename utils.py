import os
import yaml

def load_config(config_path: str, base_attr = None):
    assert os.path.exists(config_path) 
    assert os.path.isfile(config_path)
    assert config_path.endswith(".yaml") 

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    if base_attr:
        data = data[base_attr]
    
    return data

if __name__ == "__main__":
    conf = load_config("configs/training/basic.yaml", "parameters")
    print(conf)