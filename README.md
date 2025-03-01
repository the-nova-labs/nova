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
