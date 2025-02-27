import math
import random
import argparse
import asyncio
from typing import cast
from types import SimpleNamespace

import bittensor as bt
from bittensor.core.chain_data.utils import decode_metadata


def get_config():
    """
    Parse command-line arguments to set up the configuration for the wallet
    and subtensor client.
    """
    parser = argparse.ArgumentParser('Nova')
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    return bt.config(parser)


def run_model(protein: str, molecule: str) -> float:
    """
    A mock scoring function that returns a random score between 0 and 1.

    Args:
        protein (str): The current protein data.
        molecule (str): The molecule data to score against the protein.

    Returns:
        float: A random score.
    """
    return random.random()


async def get_commitments(subtensor, netuid: int, block: int = None) -> dict:
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
    # Use the provided netuid to fetch the corresponding metagraph.
    metagraph = await subtensor.metagraph(netuid)
    # Determine the block hash if a block is specified.
    block_hash = await subtensor.determine_block_hash(block) if block is not None else None

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

    while True:
        # Initialize the asynchronous subtensor client.
        subtensor = bt.async_subtensor(config=config)
        # Fetch the current metagraph for the given subnet (netuid 68).
        metagraph = await subtensor.metagraph(68)
        current_block = await subtensor.get_current_block()

        # Check if the current block marks the end of an epoch (using a 360-block interval).
        if current_block % 360 == 0:
            # Set the next commitment (here using a test string).
            await subtensor.set_commitment(
                wallet=wallet,
                netuid=68,
                data='test_prot'
            )

            # Retrieve commitments from the previous epoch.
            prev_epoch = current_block - 360 
            previous_metagraph = await subtensor.metagraph(68, block=prev_epoch)
            previous_commitments = await get_commitments(subtensor, netuid=68, block=prev_epoch)

            # Determine the current protein as that set by the validator with the highest stake.
            best_stake = -math.inf
            current_protein = None
            for hotkey, commit in previous_commitments.items():
                # Access the stake for the given hotkey from the metagraph.
                hotkey_stake = previous_metagraph.S[hotkey]
                # Choose the commitment with the highest stake and valid data.
                if hotkey_stake > best_stake and commit.data is not None:
                    best_stake = hotkey_stake
                    current_protein = commit.data

            # Retrieve the latest commitments (current epoch).
            current_commitments = await get_commitments(subtensor, netuid=68, block=current_block)

            # Identify the best molecule based on the scoring function.
            best_score = -math.inf
            best_molecule = None
            for hotkey, commit in current_commitments.items():
                # Assuming that 'commit.data' contains the necessary molecule data; adjust if needed.
                score = run_model(protein=current_protein, molecule=commit.data.get('molecule', ''))
                # If the score is higher, or equal but the block is earlier, update the best.
                if (score > best_score) or (score == best_score and best_molecule is not None and commit.block < best_molecule.block):
                    best_score = score
                    best_molecule = commit

            # Ensure a best molecule was found before setting weights.
            if best_molecule is not None:
                # Create weights where the best molecule's UID receives full weight.
                weights = [0.0] * metagraph.n
                weights[best_molecule.uid] = 1.0
                uids = list(range(metagraph.n))
                await subtensor.set_weights(
                    wallet=wallet,
                    uids=uids,
                    weights=weights,
                )
            else:
                print("No valid molecule commitment found for current epoch.")

        # Sleep briefly to prevent busy-waiting (adjust sleep time as needed).
        await asyncio.sleep(1)


if __name__ == "__main__":
    config = get_config()
    asyncio.run(main(config))
