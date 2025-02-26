import os
import time
import argparse
import traceback
from typing import Tuple
import random
import sys
import threading

import bittensor as bt
from datasets import load_dataset
from huggingface_hub import list_repo_files
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from protocol import ChallengeSynapse
from utils import get_sequence_from_protein_code
from PSICHIC.wrapper import PsichicWrapper

class Miner:
    def __init__(self):
        self.hugging_face_dataset_repo = 'Metanova/SAVI-2020'
        self.psichic_result_column_name = 'predicted_binding_affinity'
        self.chunk_size = 128
        
        self.config = self.get_config()
        self.setup_logging()
        self.setup_bittensor_objects()
        self.current_challenge_protein = None
        self.psichic_wrapper = PsichicWrapper()
        self.candidate_product = None
        self.candidate_product_score = 0
        self.best_score = 0
        self.last_submitted_product = None
        self.lock = threading.Lock()
        self.thread = None
        self.shutdown_event = threading.Event()

    def get_config(self):
        # Set up the configuration parser
        parser = argparse.ArgumentParser()
        # Adds the path to save downloaded files.
        #parser.add_argument('--savepath', default=os.path.join(BASE_DIR, 'downloaded_files'), help='Path to save downloaded files')
        # Adds override arguments for network and netuid.
        parser.add_argument('--netuid', type=int, default=309, help="The chain subnet uid.")
        # Adds subtensor specific arguments.
        bt.subtensor.add_args(parser)
        # Adds logging specific arguments.
        bt.logging.add_args(parser)
        # Adds wallet specific arguments.
        bt.wallet.add_args(parser)
        # Adds axon specific arguments.
        bt.axon.add_args(parser)
        # Parse the arguments.
        config = bt.config(parser)
        # Set up logging directory
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey_str,
                config.netuid,
                'miner',
            )
        )
        # Ensure the directories for logging and downloaded files exist.
        os.makedirs(config.full_path, exist_ok=True)
        #os.makedirs(config.savepath, exist_ok=True)

        return config

    def setup_logging(self):
        # Activate Bittensor's logging with the set configurations.
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(f"Running miner for subnet: {self.config.netuid} on network: {self.config.subtensor.network} with config:")
        bt.logging.info(self.config)

    def setup_bittensor_objects(self):
        # Initialize Bittensor miner objects
        bt.logging.info("Setting up Bittensor objects.")

        # Initialize wallet.
        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")

        # Initialize subtensor.
        self.subtensor = bt.subtensor(network="ws://localhost:9944")
        bt.logging.info(f"Subtensor: {self.subtensor}")

        # Initialize metagraph.
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(f"\nYour miner: {self.wallet} is not registered to chain connection: {self.subtensor} \nRun 'btcli register' and try again.")
            exit()
        else:
            # Each miner gets a unique identity (UID) in the network.
            self.my_subnet_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            bt.logging.info(f"Running miner on uid: {self.my_subnet_uid}")

    def blacklist_fn(self, synapse: ChallengeSynapse) -> Tuple[bool, str]:
        # Ignore requests from unrecognized entities.
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            bt.logging.trace(f'Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}')
            return True, None
        bt.logging.trace(f'Not blacklisting recognized hotkey {synapse.dendrite.hotkey}')
        return False, None
    
    def stream_random_chunk_from_dataset(self):
        # Stream random chunk from the dataset.
        files = list_repo_files(self.hugging_face_dataset_repo, repo_type='dataset')
        files = [file for file in files if file.endswith('.csv')]
        random_file = random.choice(files)
        dataset_dict = load_dataset(self.hugging_face_dataset_repo, 
                                    data_files={'train': random_file},
                                    streaming=True,
                                    )
        dataset = dataset_dict['train']
        batched = dataset.batch(self.chunk_size)
        return batched

    def run_psichic_model_loop(self):
        dataset = self.stream_random_chunk_from_dataset()
        while not self.shutdown_event.is_set():
            try:
                for chunk in dataset:
                    df = pd.DataFrame.from_dict(chunk)
                    df['product_name'] = df['product_name'].apply(lambda x: x.replace('"', ''))
                    df['product_smiles'] = df['product_smiles'].apply(lambda x: x.replace('"', ''))
                    # Run the PSICHIC model on the chunk.
                    bt.logging.debug(f'Running inference...')
                    chunk_psichic_scores = self.psichic_wrapper.run_validation(df['product_smiles'].tolist())
                    chunk_psichic_scores = chunk_psichic_scores.sort_values(by=self.psichic_result_column_name, ascending=False).reset_index(drop=True)
                    if chunk_psichic_scores[self.psichic_result_column_name].iloc[0] > self.best_score:
                        candidate_molecule = chunk_psichic_scores['Ligand'].iloc[0]
                        with self.lock:
                            self.best_score = chunk_psichic_scores[self.psichic_result_column_name].iloc[0]
                            self.candidate_product = df.loc[df['product_smiles'] == candidate_molecule, 'product_name'].iloc[0]

                            bt.logging.info(f"New best score: {self.best_score}, New candidate product: {self.candidate_product}")
            except Exception as e:
                bt.logging.error(f"Error running PSICHIC model: {e}")
                self.shutdown_event.set()      

    def respond_challenge(self, synapse: ChallengeSynapse) -> ChallengeSynapse:

        bt.logging.info(f"Received input: {synapse.target_protein}")

        # 1. Check if the challenge protein has changed.
        if self.current_challenge_protein != synapse.target_protein:

            # 1.1. Set the new challenge protein and reset best score etc.
            self.current_challenge_protein = synapse.target_protein
            bt.logging.info(f"Target protein changed to: {self.current_challenge_protein}")

            # 1.2. If the thread is running, set the shutdown event.
            if self.thread:
                if self.thread.is_alive():
                    self.shutdown_event.set()
                    bt.logging.info(f"Shutdown event set for old thread.")
    
                    # reset old values for best score, etc
                    self.candidate_product = None
                    self.candidate_product_score = 0
                    self.best_score = 0
                    self.last_submitted_product = None
                    self.shutdown_event = threading.Event()


            # 1.4. Initialize PSICHIC model for the new challenge protein.
            try:
                bt.logging.debug(f'Initializing protein sctructure')
                self.psichic_wrapper.run_challenge_start(self.current_challenge_protein)
            except Exception as e:
                bt.logging.error(f"Error initializing PSICHIC model: {e}")
                return None
            
            # 1.5. Start the PSICHIC model loop.
            try:
                self.thread = threading.Thread(target=self.run_psichic_model_loop)
                self.thread.start()
                bt.logging.debug(f'Thread started successfully')
            except Exception as e:
                bt.logging.error(f'Error initializing inference: {e}')

        # 2. Check if the candidate product has changed.
        if self.candidate_product:
            if self.candidate_product != self.last_submitted_product:
                synapse.product_name = self.candidate_product
                with self.lock:
                    self.last_submitted_product = self.candidate_product
                    #self.best_score = self.candidate_product_score
                    bt.logging.info(f"Submitted product: {self.candidate_product} with score: {self.best_score}")
            else:
                synapse.product_name = None
        else:
            synapse.product_name = None

        return synapse
        

    def setup_axon(self):
        # Build and link miner functions to the axon.
        self.axon = bt.axon(wallet=self.wallet, port=self.config.axon.port)

        # Attach functions to the axon.
        bt.logging.info(f"Attaching forward function to axon.")
        self.axon.attach(
            forward_fn=self.respond_challenge,
            blacklist_fn=self.blacklist_fn,
        )

        # Serve the axon.
        bt.logging.info(f"Serving axon on network: {self.config.subtensor.network} with netuid: {self.config.netuid}")
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        bt.logging.info(f"Axon: {self.axon}")

        # Start the axon server.
        bt.logging.info(f"Starting axon server on port: {self.config.axon.port}")
        self.axon.start()

    def run(self):
        self.setup_axon()

        # Keep the miner alive.
        bt.logging.info(f"Starting main loop")
        step = 0
        while True:
            try:
                # Periodically update our knowledge of the network graph.
                if step % 60 == 0:
                    self.metagraph.sync()
                    log = (
                        f'Block: {self.metagraph.block.item()} | '
                        f'Incentive: {self.metagraph.I[self.my_subnet_uid]} | '
                    )
                    bt.logging.info(log)
                step += 1
                time.sleep(1)

            except KeyboardInterrupt:
                self.axon.stop()
                bt.logging.success('Miner killed by keyboard interrupt.')
                break
            except Exception as e:
                bt.logging.error(traceback.format_exc())
                continue

# Run the miner.
if __name__ == "__main__":
    miner = Miner()
    miner.run()

