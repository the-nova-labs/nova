import os
from dotenv import load_dotenv
import requests

def get_smiles(product_name):

    api_key = os.environ.get("validator_api_key")
    if not api_key:
        raise ValueError("validator_api_key environment variable not set.")

    url = f"https://8vzqr9wt22.execute-api.us-east-1.amazonaws.com/dev/smiles/{product_name}"

    headers = {"x-api-key": api_key}
    
    response = requests.get(url, headers=headers)

    data = response.json()

    return data.get("smiles")

if __name__ == "__main__":

    load_dotenv()

    product_names = [
        "46BED227B506FAC0_C42C347BD57866CD_6031_UN",
        "1EB01C4D60706882_1221491DDA78048C_6031_UN",
        "does_not_exist_test",
        "1884FD2F01AFE7F4_47273D749E42DDA2_6031_UN",
    ]

    for product_name in product_names:
        smiles = get_smiles(product_name)
        if not smiles:
            print(f"{product_name}: Not found")
        else:
            print(f"{product_name}: {smiles}")


