import asyncio
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
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from my_utils import get_smiles, get_index_in_range_from_blockhash, get_protein_code_at_index, get_sequence_from_protein_code
from PSICHIC.wrapper import PsichicWrapper
from bittensor.core.chain_data.utils import decode_metadata

psichic = PsichicWrapper()

def get_config():
    """
    Parse command-line arguments to set up the configuration for the wallet
    and subtensor client.
    """
    load_dotenv()
    parser = argparse.ArgumentParser('Nova')
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)

    config = bt.config(parser)
    config.netuid = 68
    config.network = os.environ.get("SUBTENSOR_NETWORK")
    node = SubstrateInterface(url=config.network)
    config.epoch_length = node.query("SubtensorModule", "Tempo", [config.netuid]).value

    return config

async def check_registration(wallet, subtensor, netuid):
    """
    Confirm that the wallet hotkey is in the metagraph for the specified netuid.
    Logs an error and exits if it's not registered. Warns if stake is less than 1000.
    """
    metagraph = await subtensor.metagraph(netuid=netuid)
    my_hotkey_ss58 = wallet.hotkey.ss58_address

    if my_hotkey_ss58 not in metagraph.hotkeys:
        bt.logging.error(f"Hotkey {my_hotkey_ss58} is not registered on netuid {netuid}.")
        bt.logging.error("Are you sure you've registered and staked?")
        sys.exit(1) 
    
    uid = metagraph.hotkeys.index(my_hotkey_ss58)
    myStake = metagraph.S[uid]
    bt.logging.info(f"Hotkey {my_hotkey_ss58} found with UID={uid} and stake={myStake}")

    if (myStake < 1000):
        bt.logging.warning(f"Hotkey has less than 1000 stake, unable to validate")

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

async def remove_duplicate_submissions(subtensor, epoch_metagraph, last_epoch_block: int, current_block: int) -> dict:
    """
    Check for duplicate submissions in blocks and return the first occurrence of each submission.
    
    Args:
        subtensor: The subtensor client object
        epoch_metagraph: Metagraph for the epoch being analyzed
        last_epoch_block: Starting block number to check
        current_block: Ending block number to check
        
    Returns:
        dict: Mapping of valid submissions to their block/index information
    """
    bt.logging.info(f"Removing duplicate submissions from blocks {last_epoch_block} to {current_block}")
    
    valid_submissions = {}
    duplicated_hotkey_answers = set()
    
    # Get order of extrinsics in the block
    for b in range(last_epoch_block, current_block):
        seen = {}
        block_hash = await subtensor.determine_block_hash(b)
        evts = await subtensor.substrate.get_events(block_hash=block_hash)
        failed_exts = {e['extrinsic_idx'] for e in evts if e['event']['event_id'] == 'ExtrinsicFailed'}
        exts = await subtensor.substrate.get_extrinsics(block_hash=block_hash)
        
        for idx, ext in enumerate(exts):
            if (idx in failed_exts) or (ext.value["call"]["call_function"] != 'set_commitment'):
                continue
            args = {a['name']:a['value'] for a in ext.value["call"]["call_args"]}
            if args['netuid'] != config.netuid:
                continue
                
            raw = list(args['info']['fields'][0].values())[0]
            decoded_subm = str(binascii.unhexlify(raw[2:]), 'ASCII')
            
            if len(decoded_subm) < 10:
                # challenge, not submission
                continue

            hk = ext.value['address']
            uid = epoch_metagraph.hotkeys.index(hk) if hk in epoch_metagraph.hotkeys else -1
            
            # Normalize the submission for comparison
            decoded_subm_norm = decoded_subm.lower().strip()
            if decoded_subm_norm in {k.lower().strip() for k in seen.keys()}:
                bt.logging.warning(f'block {b} idx {idx}: {decoded_subm} by UID {uid} was already submitted: {seen[next(k for k in seen.keys() if k.lower().strip() == decoded_subm_norm)]}')
                duplicated_hotkey_answers.add(hk)
                continue
                
            seen[decoded_subm] = f'block {b} idx {idx} by UID {uid}'
            if decoded_subm not in valid_submissions:
                valid_submissions[decoded_subm] = {
                    'block': b,
                    'index': idx,
                    'hotkey': hk,
                    'uid': uid
                }
    
    bt.logging.warning(f"Hotkeys that submitted duplicate answers: {duplicated_hotkey_answers}")
    return valid_submissions

async def main(config):
    """
    Main routine that continuously checks for the end of an epoch to perform:
        - Setting a new commitment.
        - Retrieving past commitments.
        - Selecting the best protein/molecule pairing based on stakes and scores.
        - Setting new weights accordingly.

    Args:
        config: Configuration object for subtensor and wallet.
    """
    wallet = bt.wallet(config=config)

    # Initialize the asynchronous subtensor client.
    subtensor = bt.async_subtensor(network=config.network)
    await subtensor.initialize()

    # Check if the hotkey is registered and has at least 1000 stake.
    await check_registration(wallet, subtensor, config.netuid)

    tolerance = 3 # block tolerance window for validators to commit protein

    while True:
        # Fetch the current metagraph for the given subnet (netuid 68).
        metagraph = await subtensor.metagraph(config.netuid)
        bt.logging.debug(f'Found {metagraph.n} nodes in network')
        current_block = await subtensor.get_current_block()

        # Check if the current block marks the end of an epoch (using a 360-block interval).
        if current_block % config.epoch_length == 0:

            try:
                current_block_hash = await subtensor.determine_block_hash(current_block)
                prev_block_hash = await subtensor.determine_block_hash(current_block - 1)

                target_random_index = get_index_in_range_from_blockhash(current_block_hash, 179620)
                antitarget_random_index = get_index_in_range_from_blockhash(prev_block_hash, 179620)

                target_protein_code = get_protein_code_at_index(target_random_index)
                antitarget_protein_code = get_protein_code_at_index(antitarget_random_index)

                await subtensor.set_commitment(
                    wallet=wallet,
                    netuid=config.netuid,
                    data=f"{target_protein_code}|{antitarget_protein_code}"
                )
                bt.logging.info(f"Committed successfully target: {target_protein_code}, antitarget: {antitarget_protein_code}")

            except Exception as e:
                bt.logging.error(f"Error: {e}")
            # Retrieve commitments from the previous epoch.
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

            # Check for duplicate submissions and get valid ones
            valid_submissions = await remove_duplicate_submissions(
                subtensor,
                metagraph,
                prev_epoch,
                current_block
            )
            
            # Filter commitments based on valid submissions
            current_commitments = {
                k: v for k, v in current_commitments.items() 
                if v.data in valid_submissions and valid_submissions[v.data]['hotkey'] == k
            }
            bt.logging.debug(f"Current commitments after removing duplicates: {len(list(current_commitments.values()))}")

            # Identify the best molecule based on the scoring function.
            best_score = -math.inf
            total_commits = 0
            best_molecule = None
            for hotkey, commit in current_commitments.items():
                if current_block - commit.block <= config.epoch_length:
                    total_commits += 1
                    # Assuming that 'commit.data' contains the necessary molecule data; adjust if needed.
                    score = run_model_difference(target_protein_sequence, antitarget_protein_sequence, commit.data)
                    # If the score is higher, or equal but the block is earlier, update the best.
                    if (score > best_score) or (score == best_score and best_molecule is not None and commit.block < best_molecule.block):
                        best_score = score
                        best_molecule = commit

            # Ensure a best molecule was found before setting weights.
            if best_molecule is not None:
                try:
                    # Create weights where the best molecule's UID receives full weight.
                    weights = [0.0 for i in range(metagraph.n)]
                    print(current_block)
                    weights[best_molecule.uid] = 1.0
                    print(weights)
                    uids = list(range(metagraph.n))
                    result, message = await subtensor.set_weights(
                        wallet=wallet,
                        uids=uids,
                        weights=weights,
                        netuid=config.netuid,
                        wait_for_inclusion=True,
                        )
                    if result:
                        bt.logging.info(f"Weights set successfully: {weights}.")
                    else:
                        bt.logging.error(f"Error setting weights: {message}")
                except Exception as e:
                    bt.logging.error(f"Error setting weights: {e}")
            else:
                bt.logging.info("No valid molecule commitment found for current epoch.")

            # Sleep briefly to prevent busy-waiting (adjust sleep time as needed).
            await asyncio.sleep(1)
            
        # keep validator alive
        elif current_block % (config.epoch_length/2) == 0:
            subtensor = bt.async_subtensor(network=config.network)
            await subtensor.initialize()
            bt.logging.info("Validator reset subtensor connection.")
            await asyncio.sleep(12) # Sleep for 1 block to avoid unncessary re-connection
            
        else:
            bt.logging.info(f"Waiting for epoch to end... {config.epoch_length - (current_block % config.epoch_length)} blocks remaining.")
            await asyncio.sleep(1)


if __name__ == "__main__":
    config = get_config()
    asyncio.run(main(config))
