import os
import random
import numpy as np
import argparse
from glob import glob
import json

from src.robust_reversible_data_hiding import rrdh
from src.improved_robust_reversible_data_hiding import improved_rrdh
from src.utils import mesh_utils, util, paillier

KEY_SIZE = 512
MESSAGE_LENGTH = 256
QUANTISATION_FACTOR = 4

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/test_improved.json")
    args = parser.parse_args()

    script_dir = os.path.dirname(__file__)
    dataset_path = "./datasets/meshes/"

    config_path = args.config_path
    config_file = open(config_path, 'r')
    config = json.loads(config_file.read())
    config_file.close()

    list_methods = []
    if config["methods"] == "all":
        list_methods.extend([improved_rrdh, rrdh])
    elif "improved_rrdh" in config["methods"]:
        list_methods.append(improved_rrdh)
    elif "rrdh" in config["methods"]:
        list_methods.append(rrdh)
    else: list_methods.append(rrdh)
    
    meshes_list = []
    if config["models"] == "all":
        meshes_list = glob(dataset_path + "*.obj")
    else: 
        for m in config["models"]:
            meshes_list.extend(glob(dataset_path + m))

    
    message_bits = [k%2 for k in range(config["message_length"])]
    random.shuffle(message_bits)
    watermark = message_bits

    encryption_keys = paillier.generate_keys(config["key_size"])

    for model_path in meshes_list:
        model = mesh_utils.load_3d_model(os.path.join(script_dir, model_path))
        model_name = model_path.split('/')[-1]
        
        for method in list_methods:
            result_folder = os.path.join(script_dir, f"results/{method.__name__.split('.')[-1]}/{model_name.split(".")[0]}/")
            for f in glob(result_folder + "*"):
                os.remove(f)

            config.update({"result_folder": result_folder, "model_path": model_path, "model_name": model_name, "method_name": method.__name__})

            result = method.run(config, encryption_keys, watermark, model)

            util.write_report(result)

            print(f"Evaluation method {method.__name__.split('.')[-1]} done.")
        