#!/bin/bash
set -e

# Download model file from huggingface to avoid errors
wget -O PSICHIC/trained_weights/PDBv2020_PSICHIC/model.pt https://huggingface.co/Metanova/PSICHIC/resolve/main/model.pt

# Determine which script to run based on RUN_MODE (default to miner if not set)
if [ -z "$RUN_MODE" ]; then
    echo "RUN_MODE not set, defaulting to miner."
    SCRIPT="neurons/miner.py"
elif [ "$RUN_MODE" = "miner" ]; then
    SCRIPT="neurons/miner.py"
elif [ "$RUN_MODE" = "validator" ]; then
    SCRIPT="neurons/validator.py"
else
    echo "Unknown RUN_MODE: ${RUN_MODE}. Should be 'miner' or 'validator'."
    exit 1
fi

# Overwrite PSICHIC/runtime_config.py
if [[ -n "$DEVICE_OVERRIDE" && "$DEVICE_OVERRIDE" != "none"] ]]; then
    sed -i 's/DEVICE = "cuda:0"/DEVICE = "'"$DEVICE_OVERRIDE"'"/g' /root/PSICHIC/runtime_config.py;
fi

# Overwrite my_utils.py
if [[ -n "$VALIDATOR_API_KEY" && "$VALIDATOR_API_KEY" != "none" ]]; then
    sed -i 's/headers = {"x-api-key": api_key}/headers = {"x-api-key": "'"$VALIDATOR_API_KEY"'"}/g' /root/my_utils.py;
fi

# Build the command-line arguments array
ARGS=("$@")

# Optional flags if variables are defined in the environment
if [ -n "$WALLET_NAME" ]; then
    ARGS+=( "--wallet.name" "$WALLET_NAME" )
fi
if [ -n "$WALLET_HOTKEY" ]; then
    ARGS+=( "--wallet.hotkey" "$WALLET_HOTKEY" )
fi
if [ -n "$SUBTENSOR_NETWORK" ]; then
    ARGS+=( "--network" "$SUBTENSOR_NETWORK" )
fi

echo "Running command: python ${SCRIPT} ${ARGS[@]}"
exec python "${SCRIPT}" "${ARGS[@]}"
