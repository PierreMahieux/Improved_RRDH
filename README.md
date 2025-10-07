# An Improved Paillier-Based Reversible Watermarking Scheme for 3D Models with Reduced Complexity

The increasing adoption of cloud computing for 3D model sharing and storage necessitates robust protection mechanisms for confidentiality and ownership verification. While encryption ensures data confidentiality, watermarking techniques are required for traceability and ownership protection. This paper presents an enhanced version of an existing homomorphic encryption-based reversible watermarking scheme that employs histogram shifting and the Paillier cryptosystem for 3D models. The original method enables watermark operations in both encrypted and clear domains but suffers from high computational complexity. Our improvement refines homomorphic encryption operations while preserving the core algorithm's reversible properties. Experimental results demonstrate substantial computational time reductions of up to 99.9%  while maintaining full reversibility, security, and watermark capacity.

## Overview 

This project is developped in Python and aims to watermark 3D meshes using the homomorphic properties of the Paillier cryptosystem. We implemented the *Robust Reversible Data Hiding* method from the scientific litterature as well as as our proposition. This project allows for testing both methods on multiple 3D models through configuration files.

## Requirements

This project requires :
- gmpy2>=2.2.1
- numpy>=2.3.3

## Setup

1. Clone the repository
2. Create a virtual environment
```
python -m venv venv
```
3. Activate the environment
```
source venv/bin/activate
```
4. Install the required packages
```
pip install -r requirements.txt
```


## Running the code

To run both methods on every models in the dataset use the provided *configs/full_evaluation.json* configuration file :
```
python run_evaluation.py --config_path configs/full_evaluation.json
```
We provide 2 other configuration files that will each run a method on the *casting* object.

### Configuration files

The configuration file should be a json file with these variables :
- key_size : controls the bit size of the Paillier key, generally 1024 or 2048 bits;
- quantisation_factor: the quantisation factor to use during the preprocessing, 4 is considered enough;
- message_length: the length in bits of the message to be hidden;
- models: list of the models to watermark (*e.g.* ["casting.obj"] to watermark the *casting* object), or "all" to run the code on every object located in the *dataset/meshes* folder;
- methods: list of the methods to run (*e.g.* ["rrdh"] to run the base method), "all" to run both methods.

## License

This project is licensed under the terms of the MIT license.