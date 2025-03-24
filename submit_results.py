

import os
import requests
import bittensor as bt
from dotenv import load_dotenv
load_dotenv()


def submit_results(miner_submissions_request: dict):
    try:
        api_token = os.environ.get("API_TOKEN")
        if not api_token:
            raise ValueError("API_TOKEN environment variable not set.")

        dashboard_backend_url = os.environ.get("DASHBOARD_BACKEND_URL")
        if not dashboard_backend_url:
            raise ValueError("DASHBOARD_BACKEND_URL environment variable not set.")

        url = dashboard_backend_url + "/api/submit_results"
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, json=miner_submissions_request, headers=headers)
        if response.status_code != 200:
            bt.logging.error(f"Error submitting results: {response.status_code} {response.text}")
            return
        response_json = response.json()
        if response_json.get("success"):
            bt.logging.success(f"Results submitted successfully")
        else:
            bt.logging.error(f"Error submitting results")
    except Exception as e:
        bt.logging.error(f"Error submitting results: {e}")
