import asyncio
import json
from ast import literal_eval
import math
import os
import sys
import argparse
import binascii
from typing import cast
from types import SimpleNamespace
import bittensor as bt
from substrateinterface import SubstrateInterface
import requests
import hashlib
import subprocess
from dotenv import load_dotenv
from bittensor.core.chain_data.utils import decode_metadata

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from my_utils import (
    get_smiles, 
    get_sequence_from_protein_code, 
    submit_results
)
from PSICHIC.wrapper import PsichicWrapper
from btdr import QuicknetBittensorDrandTimelock

psichic = PsichicWrapper()
btd = QuicknetBittensorDrandTimelock()

def get_config():
    load_dotenv()
    parser = argparse.ArgumentParser('Nova')
    bt.subtensor.add_args(parser)

    config = bt.config(parser)
    config.netuid = 68
    config.network = os.environ.get("SUBTENSOR_NETWORK")
    node = SubstrateInterface(url=config.network)
    config.epoch_length = node.query("SubtensorModule", "Tempo", [config.netuid]).value

    return config

def setup_logging(config):
    """
    Configures Bittensor logging to write logs to a file named `validator.log` 
    in the same directory as this Python file
    """
    # Use the directory of this file (so validator log is in the same folder).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bt.logging(config=config, logging_dir=script_dir, record_log=True)

    bt.logging.info(f"Running validator for subnet: {config.netuid} on network: {config.subtensor.network} with config:")
    bt.logging.info(config)

def run_model(protein: str, molecule: str) -> float:
    """
    Given a protein sequence (protein) and a molecule identifier (molecule),
    retrieves its SMILES string, then uses the PsichicWrapper to produce
    a predicted binding score. Returns 0.0 if SMILES not found or if
    there's any issue with scoring.
    """

    # Initialize PSICHIC for new protein
    bt.logging.info(f'Initializing model for protein sequence: {protein}')
    try:
        psichic.run_challenge_start(protein)
        bt.logging.info('Model initialized successfully.')
    except Exception as e:
        try:
            os.system(f"wget -O {os.path.join(BASE_DIR, 'PSICHIC/trained_weights/PDBv2020_PSICHIC/model.pt')} https://huggingface.co/Metanova/PSICHIC/resolve/main/model.pt")
            psichic.run_challenge_start(protein)
            bt.logging.info('Model initialized successfully.')
        except Exception as e:
            bt.logging.error(f'Error initializing model: {e}')


    smiles = get_smiles(molecule)
    if not smiles:
        bt.logging.debug(f"Could not retrieve SMILES for '{molecule}', returning score of 0.0.")
        return 0.0

    results_df = psichic.run_validation([smiles])  # returns a DataFrame
    if results_df.empty:
        bt.logging.warning("Psichic returned an empty DataFrame, returning 0.0.")
        return 0.0

    predicted_score = results_df.iloc[0]['predicted_binding_affinity']
    if predicted_score is None:
        bt.logging.warning("No 'predicted_binding_affinity' found, returning 0.0.")
        return 0.0

    return float(predicted_score)

def run_model_difference(target_sequence: str, antitarget_sequence: str, molecule: str) -> float:
    """
    Compute final_score = binding_affinity(target) - binding_affinity(anti-target)
    """
    s_target = run_model(protein=target_sequence, molecule=molecule)
    s_anti   = run_model(protein=antitarget_sequence, molecule=molecule)
    return s_target - s_anti


async def get_commitments(subtensor, metagraph, block_hash: str, netuid: int) -> dict:
    """
    Retrieve commitments for all miners on a given subnet (netuid) at a specific block.

    Args:
        subtensor: The subtensor client object.
        netuid (int): The network ID.
        block (int, optional): The block number to query. Defaults to None.

    Returns:
        dict: A mapping from hotkey to a SimpleNamespace containing uid, hotkey,
              block, and decoded commitment data.
    """

    # Gather commitment queries for all validators (hotkeys) concurrently.
    commits = await asyncio.gather(*[
        subtensor.substrate.query(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[netuid, hotkey],
            block_hash=block_hash,
        ) for hotkey in metagraph.hotkeys
    ])

    # Process the results and build a dictionary with additional metadata.
    result = {}
    for uid, hotkey in enumerate(metagraph.hotkeys):
        commit = cast(dict, commits[uid])
        if commit:
            result[hotkey] = SimpleNamespace(
                uid=uid,
                hotkey=hotkey,
                block=commit['block'],
                data=decode_metadata(commit)
            )
    return result

def tuple_safe_eval(input_str: str) -> tuple:
    # Limit input size to prevent overly large inputs.
    if len(input_str) > 1024:
        raise ValueError("Input exceeds allowed size")
    
    try:
        # Safely evaluate the input string as a Python literal.
        result = literal_eval(input_str)
    except (SyntaxError, ValueError) as e:
        bt.logging.error(f"Input is not a valid literal: {e}")
        return None

    # Check that the result is a tuple with exactly two elements.
    if not (isinstance(result, tuple) and len(result) == 2):
        bt.logging.error("Expected a tuple with exactly two elements")
        return None

    # Verify that the first element is an int.
    if not isinstance(result[0], int):
        bt.logging.error("First element must be an int")
        return None
    
    # Verify that the second element is a bytes object.
    if not isinstance(result[1], bytes):
        bt.logging.error("Second element must be a bytes object")
        return None
    
    return result

def decrypt_submissions(current_commitments: dict, headers: dict = {"Range": "bytes=0-1024"}) -> dict:
    """
    Decrypts submissions from validators by fetching encrypted content from GitHub URLs and decrypting them.

    Args:
        current_commitments (dict): A dictionary of miner commitments where each value contains:
            - uid: Miner's unique identifier
            - data: GitHub URL path containing the encrypted submission
            - Other commitment metadata
        headers (dict, optional): HTTP request headers for fetching content. 
            Defaults to {"Range": "bytes=0-1024"} to limit response size.

    Returns:
        dict: A dictionary of decrypted submissions mapped by validator UIDs.
            Empty if no valid submissions were found or decryption failed.

    Note:
        - Only processes commitments where data contains a '/' (indicating a GitHub URL)
        - Uses btd.decrypt_dict for decryption of the fetched submissions
        - Logs errors for failed HTTP requests and submission counts
    """
    encrypted_submissions = {}
    for commit in current_commitments.values():
        if '/' not in commit.data:
            continue
        try:
            full_url = f"https://raw.githubusercontent.com/{commit.data}"
            response = requests.get(full_url, headers=headers)
            if response.status_code in [200, 206]:
                encrypted_content = response.content
                content_hash = hashlib.sha256(encrypted_content.decode('utf-8').encode('utf-8')).hexdigest()[:20]

                # Disregard any submissions that don't match the expected filename
                if not full_url.endswith(f'/{content_hash}.txt'):
                    bt.logging.error(f"Filename for {commit.uid} is not compatible with expected content hash")
                    continue
                encrypted_content = encrypted_content.decode('utf-8', errors='replace')

                # Safely evaluate the input string as a Python literal.
                encrypted_content = tuple_safe_eval(encrypted_content)
                if encrypted_content is None:
                    bt.logging.error(f"Encrypted content for {commit.uid} is not a tuple")
                    continue
                encrypted_submissions[commit.uid] = (encrypted_content[0], encrypted_content[1])
            else:
                bt.logging.error(f"Error fetching encrypted submission: {response.status_code}")
                bt.logging.error(f"uid: {commit.uid}, commited data: {commit.data}")
                continue
        except Exception as e:
            bt.logging.error(f"Error decrypting submission: {e}")
            continue

    bt.logging.info(f"Encrypted submissions: {len(encrypted_submissions)}")
    decrypted_submissions = btd.decrypt_dict(encrypted_submissions)
    bt.logging.info(f"Decrypted submissions: {len(decrypted_submissions)}")
            
    return decrypted_submissions


async def main(config):
    """
    Main routine that continuously checks for the end of an epoch to perform:
        - Setting a new commitment.
        - Retrieving past commitments.
        - Selecting the best protein/molecule pairing based on stakes and scores.
        - Setting new weights accordingly.
    """

    # Initialize the asynchronous subtensor client.
    subtensor = bt.async_subtensor(network=config.network)
    await subtensor.initialize()

    tolerance = 3 # block tolerance window for validators to commit protein

    while True:
        # Fetch the current metagraph for the given subnet (netuid 68).
        metagraph = await subtensor.metagraph(config.netuid)
        current_block = await subtensor.get_current_block()

        # Check if the current block marks the end of an epoch (using a 360-block interval).
        if current_block % config.epoch_length == 0:            # Retrieve commitments from the previous epoch.
            bt.logging.info(f"Committing for epoch {current_block // config.epoch_length - 1}")
            prev_epoch = current_block - config.epoch_length
            best_stake = -math.inf
            current_protein = None

            block_to_check = prev_epoch
            block_hash_to_check = await subtensor.determine_block_hash(block_to_check + tolerance)  
            epoch_metagraph = await subtensor.metagraph(config.netuid, block=block_to_check + tolerance)
            epoch_commitments = await get_commitments(subtensor, epoch_metagraph, block_hash_to_check, netuid=config.netuid)
            epoch_commitments = {k: v for k, v in epoch_commitments.items() if current_block - v.block <= (config.epoch_length + tolerance)}
            
            high_stake_protein_commitment = max(
                epoch_commitments.values(),
                key=lambda commit: epoch_metagraph.S[commit.uid],
                default=None
            )
            if not high_stake_protein_commitment:
                bt.logging.error("Error getting current protein commitment.")
                current_protein = None
                continue

            protein_codes = high_stake_protein_commitment.data.split('|')
            target_protein_code = protein_codes[0]
            antitarget_protein_code = protein_codes[1]
            bt.logging.info(f"Current target protein: {target_protein_code}, antitarget: {antitarget_protein_code}")

            target_protein_sequence = get_sequence_from_protein_code(target_protein_code)
            antitarget_protein_sequence = get_sequence_from_protein_code(antitarget_protein_code)

            # Retrieve the latest commitments (current epoch).
            current_block_hash = await subtensor.determine_block_hash(current_block)
            current_commitments = await get_commitments(subtensor, metagraph, current_block_hash, netuid=config.netuid)
            bt.logging.debug(f"Current commitments: {len(list(current_commitments.values()))}")

            # Decrypt submissions
            decrypted_submissions = decrypt_submissions(current_commitments)

            # Identify the best molecule based on the scoring function.
            best_score = -math.inf
            total_commits = 0
            best_molecule = None
            epoch_number = current_block // config.epoch_length - 1
            competition = {
                "epoch_number": epoch_number,
                "target_protein": target_protein_code,
                "anti_target_protein": antitarget_protein_code,
            }
            submissions = []
            for hotkey, commit in current_commitments.items():
                if current_block - commit.block <= config.epoch_length:
                    # Find the decrypted submission for the current commitment
                    try:    
                        molecule = decrypted_submissions[commit.uid]
                        total_commits += 1
                    except Exception as e:
                        bt.logging.error(f"Decrypted submission for {commit.uid} not found: {e}")
                        continue
                    try:
                        score = run_model_difference(target_protein_sequence, antitarget_protein_sequence, molecule)
                        score = round(score, 3)
                    except Exception as e:
                        bt.logging.error(f"Error scoring molecule {molecule}: {e}")
                        score = -math.inf
                    submissions.append({
                        "neuron": {
                            "hotkey": hotkey,
                        },
                        "block_number": commit.block,
                        "molecule": molecule,
                        "score": score,
                    })
            
            submit_results({
                "competition": competition,
                "submissions": submissions,
            })
            
            try:
                save_file_path = f"results/submissions_epoch_{epoch_number}.json"
                submissions.sort(key=lambda x: (x['score'], -x['block_number']), reverse=True)
                with open(save_file_path, 'w') as f:
                    json.dump({
                        "competition": competition,
                        "submissions": submissions,
                    }, f, indent=4)
            except Exception as e:
                bt.logging.error(f"Error saving submissions: {e}")
            await asyncio.sleep(1)
        # keep validator alive
        elif current_block % (config.epoch_length/2) == 0:
            subtensor = bt.async_subtensor(network=config.network)
            await subtensor.initialize()
            bt.logging.info("Validator reset subtensor connection.")
            await asyncio.sleep(12) # Sleep for 1 block to avoid unncessary re-connection

        else:
            await asyncio.sleep(1)


if __name__ == "__main__":
    config = get_config()
    setup_logging(config)
    asyncio.run(main(config))
