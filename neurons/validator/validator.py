import os
import random
import argparse
import traceback
import bittensor as bt
import time
import sys
import csv 

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from protocol import ChallengeSynapse
from utils import get_smiles, get_active_challenge
from neurons.validator.db_manager import DBManager
from PSICHIC.wrapper import PsichicWrapper
from substrateinterface import SubstrateInterface


class Validator:
    def __init__(self):
        self.config = self.get_config()
        self.setup_logging()
        self.setup_bittensor_objects()
        self.last_update = 0
        self.my_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.scores = [1.0] * len(self.metagraph.S)
        self.last_update = 0
        self.current_block = 0
        self.tempo = self.node_query('SubtensorModule', 'Tempo', [self.config.netuid])
        self.moving_avg_scores = [1.0] * len(self.metagraph.S)
        self.alpha = 0.1
        self.node = SubstrateInterface(url=self.config.subtensor.chain_endpoint)
        self.db = DBManager()
        self.psichic = PsichicWrapper()
        self.finalized_challenge = False

    def get_config(self):
        # Set up the configuration parser.
        parser = argparse.ArgumentParser()
        # TODO: Add your custom validator arguments to the parser.
        parser.add_argument('--custom', default='my_custom_value', help='Adds a custom value to the parser.')
        # Adds override arguments for network and netuid.
        parser.add_argument('--netuid', type=int, default=1, help="The chain subnet uid.")
        # Adds subtensor specific arguments.
        bt.subtensor.add_args(parser)
        # Adds logging specific arguments.
        bt.logging.add_args(parser)
        # Adds wallet specific arguments.
        bt.wallet.add_args(parser)
        # Parse the config.
        config = bt.config(parser)
        # Set up logging directory.
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey_str,
                config.netuid,
                'validator',
            )
        )
        # Ensure the logging directory exists.
        os.makedirs(config.full_path, exist_ok=True)
        return config

    def setup_logging(self):
        # Set up logging.
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(f"Running validator for subnet: {self.config.netuid} on network: {self.config.subtensor.network} with config:")
        bt.logging.info(self.config)

    def setup_bittensor_objects(self):
        # Build Bittensor validator objects.
        bt.logging.info("Setting up Bittensor objects.")

        # Initialize wallet.
        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")

        # Initialize subtensor.
        self.subtensor = bt.subtensor(network="test")
        bt.logging.info(f"Subtensor: {self.subtensor}")

        # Initialize dendrite.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Initialize metagraph.
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        # Connect the validator to the network.
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(f"\nYour validator: {self.wallet} is not registered to chain connection: {self.subtensor} \nRun 'btcli register' and try again.")
            exit()
        else:
            # Each validator gets a unique identity (UID) in the network.
            self.my_subnet_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            bt.logging.info(f"Running validator on uid: {self.my_subnet_uid}")

        # Set up initial scoring weights for validation.
        bt.logging.info("Building validation weights.")
        self.scores = [1.0] * len(self.metagraph.S)
        bt.logging.info(f"Weights: {self.scores}")

    def node_query(self, module, method, params):
        try:
            result = self.node.query(module, method, params).value

        except Exception:
            # reinitilize node
            self.node = SubstrateInterface(url=self.config.subtensor.chain_endpoint)
            result = self.node.query(module, method, params).value
        
        return result
    
    
    def query_and_store_challenge_results(self, target_protein):
        """
        1) Create a ChallengeSynapse with a given target_protein.
        2) Query all miners.
        3) Convert each product_name to SMILES.
        4) Score them with PsichicWrapper.
        5) Store in the local DB if it's the miner's best so far.
        """

        # Build a ChallengeSynapse with a hardcoded protein (for now).
        synapse = ChallengeSynapse(
            target_protein = target_protein
        )

        # Get validator's hotkey to avoid querying itself
        my_hotkey = self.wallet.hotkey.ss58_address
        all_axons = self.metagraph.axons

        # Filter out your own axon so that you never query yourself.
        filtered_axons = [axon for axon in all_axons if axon.hotkey != my_hotkey]

        # Query all miners
        responses = self.dendrite.query(
            axons=filtered_axons,
            synapse=synapse,
            timeout=12
        )

        if responses is None:
            bt.logging.warning("No responses returned from the dendrite query.")
            return

        # Collect (miner_uid -> product_name)
        miner_submissions = {}
        for i, resp in enumerate(responses):
            if resp is not None and resp.product_name is not None:
                miner_hotkey = filtered_axons[i].hotkey
                miner_uid = self.metagraph.hotkeys.index(miner_hotkey)
                miner_submissions[miner_uid] = resp.product_name


        bt.logging.info(f"Collected product_name from {len(miner_submissions)} miners: {miner_submissions}")

        #Convert each product_name -> SMILES
        uid_to_smiles = {}
        for uid, pname in miner_submissions.items():
            smiles = get_smiles(pname)
            if smiles:
                uid_to_smiles[uid] = smiles
            else:
                bt.logging.warning(f"product_name {pname} (miner uid {uid}) not found or invalid in DB.")

        # If no valid SMILES, nothing to score.
        if not uid_to_smiles:
            bt.logging.info("No valid SMILES to score.")
            return

        #  Score all SMILES via PsichicWrapper
        smiles_list = list(uid_to_smiles.values())
        self.psichic.run_challenge_start(synapse.target_protein)
        results_df = self.psichic.run_validation(smiles_list)
        # Returns a DataFrame with columns like ['Protein', 'Ligand', 'predicted_binding_affinity']
        bt.logging.info(f"Scoring results:\n{results_df}")

        # Map each row back to the correct miner (by matching SMILES)
        for idx, row in results_df.iterrows():
            ligand_smiles = row['Ligand'] 
            predicted_score = row['predicted_binding_affinity'] 
            matched_uid = None
            for uid, smi in uid_to_smiles.items():
                if smi == ligand_smiles:
                    matched_uid = uid
                    break

            if matched_uid is not None:
                # Update each miner’s best submission in the DB
                self.db.update_best_score(
                    miner_uid = matched_uid,
                    miner_hotkey = miner_hotkey,
                    smiles = ligand_smiles,
                    score = float(predicted_score)
                )

        # Award bounty to the miner with the highest score
        self.db.award_bounty(1)

        bt.logging.info("Done updating best scores in DB.")

    def finalize_challenge(self, challenge_id: int, protein_code: str):
        """
        Finalizes a challenge by:
         1. Gathering the final scoreboard from the local DB.
         2. Setting on-chain weights.
         3. Exporting the scoreboard to a CSV.
         4. TODO: Upload the CSV to S3.
        """
        bt.logging.info(f"Finalizing challenge {challenge_id}...")

        scoreboard = self.get_final_scoreboard()
        if not scoreboard:
            bt.logging.info("No entries found in scoreboard; skipping weights and export.")
            return

        self.set_challenge_weights(scoreboard)

        csv_path = self.export_scoreboard_to_csv(challenge_id, scoreboard)
        bt.logging.info(f"Exported final scoreboard to {csv_path}")

        #Upload to S3

    def get_final_scoreboard(self):
        """
        Retrieves the best scores and bounty points for each miner from the local DB and
        returns a dict scoreboard
        """
        best_scores_data = self.db.get_all_best_scores()  # Add this method to DBManager
        scoreboard = {}
        for row in best_scores_data:
            uid, hotkey, best_smiles, best_score, bounty_points = row
            scoreboard[uid] = {
                'hotkey': hotkey,
                'best_smiles': best_smiles,
                'best_score': best_score,
                'bounty_points': bounty_points
            }
        return scoreboard

    def set_challenge_weights(self, scoreboard: dict):
        """
        Normalizes each miner's best_score and calls Bittensor to set weights on-chain.
        """
        max_score = max(entry['best_score'] for entry in scoreboard.values())
        if max_score <= 0:
            bt.logging.info("Max score <= 0; skipping on-chain weight update.")
            return

        total_miners = len(self.metagraph.hotkeys)
        weights = []
        for uid in range(total_miners):
            entry = scoreboard.get(uid)
            if entry:
                normalized = entry['best_score'] / max_score
            else:
                normalized = 0.0
            weights.append(normalized)

        try:
            self.subtensor.set_weights(
                netuid=self.config.netuid,
                uids=list(range(total_miners)),
                weights=weights,
                wallet=self.wallet
            )
            bt.logging.info("Successfully set weights on chain.")
        except Exception as e:
            bt.logging.error(f"Error calling set_weights: {e}")

    def export_scoreboard_to_csv(self, challenge_id: int, scoreboard: dict) -> str:
        """
        Writes scoreboard data (including bounty_points) to a CSV file and returns the local file path.
        """
        csv_dir = os.path.join(BASE_DIR, "temp_data")
        os.makedirs(csv_dir, exist_ok=True)

        filename = f"challenge_{challenge_id}_scoreboard.csv"
        csv_path = os.path.join(csv_dir, filename)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["miner_uid", "hotkey", "best_score", "best_smiles", "bounty_points"])
            for uid, data in scoreboard.items():
                writer.writerow([
                    uid,
                    data.get('hotkey', ''),
                    data.get('best_score', 0),
                    data.get('best_smiles', ''),
                    data.get('bounty_points', 0)
                ])

        return csv_path


    def run(self):
        # The Main Validation Loop.
        bt.logging.info("Starting validator loop.")
        while True:
            try:
                # Retrieve the challenge (if any) that is not finished
                challenge = get_active_challenge()

                if challenge is None:
                    bt.logging.info("No active challenge found. Sleeping...")
                    time.sleep(10)
                    continue

                status = challenge["status"]
                challenge_id = challenge["challenge_id"]
                protein_code = challenge["target_protein"]

                if status == "in_progress":
                    self.finalized_challenge = False
                    self.query_and_store_challenge_results(protein_code)

                elif status == "finalizing":
                    if not self.finalized_challenge:
                        self.finalize_challenge(challenge_id, protein_code)
                        self.finalized_challenge = True
                    else:
                        bt.logging.info(f"Challenge {challenge_id} is already finalized. No action taken.")

                elif status == "finished":
                    bt.logging.info(f"Challenge {challenge_id} is finished. No action taken.")
                
                time.sleep(10)


            except RuntimeError as e:
                bt.logging.error(e)
                traceback.print_exc()

            except KeyboardInterrupt:
                bt.logging.success("Keyboard interrupt detected. Exiting validator.")
                exit()

# Run the validator.
if __name__ == "__main__":
    validator = Validator()
    validator.run()
