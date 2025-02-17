import requests
import os
import sys
import json
from dotenv import load_dotenv
import psycopg2

load_dotenv()

def get_smiles(product_name):

    api_key = os.environ.get("validator_api_key")
    if not api_key:
        raise ValueError("validator_api_key environment variable not set.")

    url = f"https://8vzqr9wt22.execute-api.us-east-1.amazonaws.com/dev/smiles/{product_name}"

    headers = {"x-api-key": api_key}
    
    response = requests.get(url, headers=headers)

    data = response.json()

    return data.get("smiles")

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
    

def get_in_progress_target_protein():
    """
    Returns target_protein from the first challenge with status='in_progress',
    or None if none is found.
    """
    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASS")
    db_port = os.getenv("DB_PORT") 

    conn = None
    try:
        conn = psycopg2.connect(
            host=db_host,
            dbname=db_name,
            user=db_user,
            password=db_pass,
            port=db_port
        )
        cur = conn.cursor()
        cur.execute("""
            SELECT target_protein
            FROM challenges
            WHERE status = 'in_progress'
            LIMIT 1
        """)
        row = cur.fetchone()
        cur.close()
        conn.close()
        
        if row:
            protein_code = row[0]
            sequence = get_sequence_from_protein_code(protein_code)
            return sequence  # returns the amino-acid sequence
        else:
            return None
        
    except Exception as e:
        if conn:
            conn.close()
        print(f"Error querying in-progress challenge: {e}")
        return None

if __name__ == '__main__':
    protein_codes = [
            'P21554',
            'P28223',
            'doesnt_exist',
            'P43220'
            ]

    for protein_code in protein_codes:
        sequence = get_sequence_from_protein_code(protein_code)

        if not sequence:
            print(f'{protein_code}: Not found')
        else:
            print(f'{protein_code}: {sequence}')



