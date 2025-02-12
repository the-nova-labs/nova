import requests
import sys
import json

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



