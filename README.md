# NOVA - SN68

## High-throughput ML-driven drug screening.

### Accelerating drug discovery, powered by Bittensor.

NOVA harnesses global compute and collective intelligence to navigate huge unexplored chemical spaces, uncovering breakthrough compounds at a fraction of the cost and time.

## Project Structure

```
nova/
├── neurons/           # Core components
│   ├── miner.py      # Miner implementation
│   └── validator.py  # Validator implementation
├── requirements/      # Dependency specifications
│   ├── requirements_cpu.txt
│   └── requiremets_cu124.txt
├── PSICHIC/          # PSICHIC integration
├── install_deps_cpu.sh
├── install_deps_cu124.sh
└── my_utils.py       # Utility functions
```

## System Requirements

- Ubuntu 24.04 LTS (recommended)
- Python 3.12
- CUDA 12.4 (for GPU support)
- Sufficient RAM for ML model operations
- Internet connection for network participation

## Installation and Running

1. Clone the repository:
```bash
git clone <repository-url>
cd nova
```

2. Prepare your .env file:
```
VALIDATOR_API_KEY=<your_api_key> # validators only

# For validators, we highly recommend running a local node!
SUBTENSOR_NETWORK="wss://archive.chain.opentensor.ai:443 

DEVICE_OVERRIDE="cpu" #or none to run on GPU
```

3. Install dependencies:
   - For CPU:
   ```bash
   bash install_deps_cpu.sh
   ```
   - For CUDA 12.4:
   ```bash
   bash install_deps_cu124.sh
   ```

4. Run:
```bash
# Activate your virtual environment:
source .venv/bin/activate

# Run your script:
# miner:
python3 neurons/miner.py --wallet.name <your_wallet> --wallet.hotkey <your_hotkey> --logging.debug

# validator:
python3 neurons/validator.py --wallet.name <your_wallet> --wallet.hotkey <your_hotkey> --logging.debug
```

## Configuration

The project uses several configuration files:
- `.env`: Environment variables and API keys
- `requirements/`: Dependency specifications for different environments
- Command-line arguments for runtime configuration

## Development Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install development dependencies:
```bash
pip install -r requirements/requirements_cpu.txt  # or cu124 for GPU
```

## For Validators

DM the NOVA team to obtain an API key.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Support

For support, please open an issue in the repository or contact the NOVA team.
