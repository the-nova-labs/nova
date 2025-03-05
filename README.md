# NOVA - SN68
## High-throughput ML-driven drug screening.
### Accelerating drug discovery, powered by Bittensor.
NOVA harnesses global compute and collective intelligence to navigate huge unexplored chemical spaces, uncovering breakthrough compounds at a fraction of the cost and time.


## Installation and running
> Recommended: Ubuntu 24.04 LTS, Python 3.12

1. Prepare your .env file:
```
    VALIDATOR_API_KEY=<your_api_key>   # validators only
    HOST_HOME=/home/<your_username>
    WALLET_NAME=<your_wallet_name>
    WALLET_HOTKEY=<yout_hotkey_name>
    SUBTENSOR_NETWORK=wss://archive.chain.opentensor.ai:443  # or your local node
    RUN_MODE=validator     #or miner
    DEVICE_OVERRIDE=cpu    #or none to run on GPU
```
2. Build image:
```
    docker build -t nova .
```
This will install all dependencies and set everything up for you to run either as a miner or a validator. Updates to this repo will be applies automatically with Watchtower.

3. Running:
`docker-compose up -d`
	
 4. To view logs:
`docker logs -f nova`
	
   


### For validators: 
 DM the NOVA team to obtain an API key.

