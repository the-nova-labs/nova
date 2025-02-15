import os
import random
import argparse
import traceback
import bittensor as bt
import time

from protocol import ChallengeSynapse
from utils import get_smiles
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
    
    def query_and_store_challenge_results(self):
        """
        1) Create a ChallengeSynapse with a fixed target_protein (for now).
        2) Query all miners.
        3) Convert each product_name to SMILES.
        4) Score them with PsichicWrapper.
        5) Store in the local DB if it's the miner's best so far.
        """

        # Build a ChallengeSynapse with a hardcoded protein (for now).
        synapse = ChallengeSynapse(
            target_protein = "MKSILDGLADTTFRTITTDLLYVGSNDIQYEDIKGDMASKLGYFPQKSPLTSFRGSPFQEKMTAGDNPQLVPADQVNITEFYNKSLSSFKENEENIQCGENFMDIECFMVLNPSQQLAIAVLSLTLGTFTVLENLLVLCVILHSRSLRCRPSYHFIGSLAVADLLGSVIFVYSFIDFHVFHRKDSRNVFLFKLGGVTASFTASVGSLFLTAIDRYISIHRPLAYKRIVTRPKAVVAFCLMWTIAIVIAVLPLLGWNCEKLQSVCSDILPHIDETYLMLWIGVTSVLLLFIVYAYMYILWKAHSHAVRMIQRGAQKSIIIHTSEDGKVQVTRPDQARMDIRLAKTLVLILVVLIICWGPLLAIMVYDVFGKMNKLIKTVFAFCSMLCLLNSTVNPIIYALRSKDLRHAFRSMFPSCGGTAQPLDNSMGDSDCLHKHANNAASVHRAAESCIKSTVKIAKVTMSVSTDTSAEAL"  # Example protein
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
                miner_uid = self.metagraph.uids[i]
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
        results_df = self.psichic.run_validation(smiles_list)
        # Typically returns a DataFrame with columns like ['Protein', 'Ligand', 'predicted_score']
        bt.logging.info(f"Scoring results:\n{results_df}")

        # Map each row back to the correct miner (by matching SMILES)
        for idx, row in results_df.iterrows():
            ligand_smiles = row['Ligand'] 
            predicted_score = row['predicted_score'] 
            matched_uid = None
            for uid, smi in uid_to_smiles.items():
                if smi == ligand_smiles:
                    matched_uid = uid
                    break

            if matched_uid is not None:
                # Update each miner’s best submission in the DB
                self.db.update_best_score(
                    miner_uid = matched_uid,
                    smiles = ligand_smiles,
                    score = float(predicted_score)
                )

        bt.logging.info("Done updating best scores in DB.")


    def run(self):
        # The Main Validation Loop.
        bt.logging.info("Starting validator loop.")
        while True:
            try:
                self.query_and_store_challenge_results()
                time.sleep(10)  # wait between queries

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
