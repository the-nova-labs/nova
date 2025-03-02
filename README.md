# NOVA - SN68
## High-throughput ML-driven drug screening.
### Accelerating drug discovery, powered by Bittensor.
NOVA harnesses global compute and collective intelligence to navigate huge unexplored chemical spaces, uncovering breakthrough compounds at a fraction of the cost and time.


## Installation
> Recommended: Ubuntu 24.04 LTS, Python 3.12

    python3.12 -m venv <your_env>
    source <your_env>/bin/activate
    pip install -r requirements1.txt
    pip install -r requirements2.txt


## Running
### For miners:

    python3 neurons/miner.py --wallet.name <your_wallet> --wallet.hotkey <your_hotkey> --logging.debug

### For validators: 

1. DM the NOVA team to obtain an API key.
2. Set validator_api_key=<your_api_key> in your .env file (create it if it doesn’t exist).

Once that is done run:
```
python3 neurons/validator.py --wallet.name <your_wallet> --wallet.hotkey <your_hotkey> --logging.debug
```

## Configuration for CPU
If you're running on a CPU-only system (no GPU), you will need to modify the PSICHIC/runtime_config.py file:
```
DEVICE = 'cpu'
```

## Troubleshooting
If you get the error `Error running PSICHIC model: 'PsichicWrapper' object has no attribute 'protein_dict'`, it means the weights are not downloaded correctly on your local machine.
You can download the trained weights from [our Hugging Face repo](https://huggingface.co/Metanova/PSICHIC/tree/main) 
and place it in the `PSICHIC/trained_weights/PDBv2020_PSICHIC` folder by running:
```
wget -O PSICHIC/trained_weights/PDBv2020_PSICHIC/model.pt \ 
  https://huggingface.co/Metanova/PSICHIC/resolve/main/model.pt
```
