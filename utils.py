import requests
import os
import sys
import json
from dotenv import load_dotenv
import psycopg2
import bittensor as bt

load_dotenv(override=True)

def get_smiles(product_name):

    api_key = os.environ.get("validator_api_key")
    if not api_key:
        raise ValueError("validator_api_key environment variable not set.")

    url = f"https://8vzqr9wt22.execute-api.us-east-1.amazonaws.com/dev/smiles/{product_name}"

    headers = {"x-api-key": api_key}
    
    response = requests.get(url, headers=headers)

    data = response.json()

    return data.get("smiles")

def get_random_protein():
    api_key = os.environ.get("validator_api_key")
    if not api_key:
        raise ValueError("validator_api_key environment variable not set contact nova team for api key.")

    url = "https://rvhs77j663.execute-api.us-east-1.amazonaws.com/prod/random-protein-of-interest"
    headers = {"x-api-key": api_key}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise RuntimeError(f"API call failed: {response.status_code} {response.text}")

    data = response.json()

    if "body" in data:
        inner_body_str = data["body"]  # e.g. '{"uniprot_code": "A0S183", "protein_sequence": "..."}'
        try:
            inner_data = json.loads(inner_body_str) 
            return inner_data.get("uniprot_code")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Could not parse body as JSON: {e}")
    else:
        return None  # or raise an error if "body" is missing


def get_sequence_from_protein_code(protein_code:str) -> str:

    url = f"https://rest.uniprot.org/uniprotkb/{protein_code}.fasta"
    response = requests.get(url)

    if response.status_code != 200:
        return None
    else:
        lines = response.text.splitlines()
        sequence_lines = [line.strip() for line in lines if not line.startswith('>')]
        amino_acid_sequence = ''.join(sequence_lines)
        return amino_acid_sequence
    
def get_active_challenge():

    api_key = os.environ.get("validator_api_key")
    if not api_key:
        raise ValueError("validator_api_key environment variable not set.")

    url = f"https://hbsndyd1td.execute-api.us-east-1.amazonaws.com/prod/active_challenge"

    headers = {"x-api-key": api_key}
    
    response = requests.get(url, headers=headers)

    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            # Expected: { "data": { "challenge_id", "target_protein", "status" } }
            payload = response.json()
            data = payload.get("data")
            if not data:
                bt.logging.warning("No 'data' field in the API response.")
                return None

            challenge_id = data.get("challenge_id")
            protein_code = data.get("target_protein")
            status = data.get("status")

            if not protein_code:
                bt.logging.warning("No 'target_protein' in the API response data.")
                return None

            full_sequence = get_sequence_from_protein_code(protein_code)
            if not full_sequence:
                bt.logging.warning(f"Could not retrieve sequence for code '{protein_code}'")
                return None

            # Return the final dict, matching the old structure
            return {
                "challenge_id": challenge_id,
                "target_protein": full_sequence,
                "status": status
            }

        elif response.status_code == 404:
            bt.logging.info("No active challenge found (404).")
            return None
        else:
            bt.logging.warning(f"Unexpected HTTP {response.status_code} => {response.text}")
            return None

    except Exception as e:
        bt.logging.warning(f"Error calling active_challenge API: {e}", exc_info=True)
        return None
