import os
import random
import numpy as np
from glob import glob

from src.robust_reversible_data_hiding import rrdh
from src.improved_robust_reversible_data_hiding import improved_rrdh
from src.utils import mesh_utils, util, paillier

KEY_SIZE = 512
MESSAGE_LENGTH = 256
QUANTISATION_FACTOR = 4

if __name__ == "__main__":
    list_methods = [rrdh, improved_rrdh]
    script_dir = os.path.dirname(__file__)

    dataset_path = "./datasets/meshes/"
    meshes_list = glob(dataset_path + "*.obj")
    
    
    message_bits = [np.random(0, 2) for _ in range(MESSAGE_LENGTH)]
    watermarks = (message_bits)

    encryption_keys = paillier.generate_keys(KEY_SIZE)

    for model_path in meshes_list:
        model = mesh_utils.load_3d_model(os.path.join(script_dir, model_path))
        model_name = model_path.split('/')[-1]
        
        for method in list_methods:
            result_folder = os.path.join(script_dir, f"results/{method.__name__.split('.')[-1]}/{model_name.split(".")[0]}/")
            for f in glob(result_folder + "*"):
                os.remove(f)

            config = {"key_size": KEY_SIZE, "quantisation_factor": QUANTISATION_FACTOR, "result_folder": result_folder, "model_path": model_path, "model_name": model_name, "message_length": MESSAGE_LENGTH, "method_name": method.__name__}

            result = method.run(config, encryption_keys, watermarks, model)

            util.write_report(result)

            print(f"Evaluation method {method.__name__.split('.')[-1]} done.")
        