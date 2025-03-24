#!/usr/bin/env python3
# credits: Rhef
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, Optional
import asyncio
import base64
import hashlib
import logging
import secrets
import time

from cryptography.fernet import Fernet
import requests
import timelock
import bittensor as bt

logger = logging.getLogger(__name__)


class TooEarly(RuntimeError):
    pass


class DrandClient:
    """Class for Drand-based timelock encryption and decryption."""

    RETRY_LIMIT = 30
    RETRY_BACKOFF_S = 2

    def __init__(self, url):
        """Initialize a requests session for better performance."""
        self.session: requests.Session = requests.Session()
        self.url = url

    def get(self, round_number: int, retry_if_too_early=False) -> str:
        """Fetch the randomness for a given round, using cache to prevent duplicate requests."""
        a = 0
        while a <= self.RETRY_LIMIT:
            a += 1
            response: requests.Response = self.session.get(f"{self.url}/public/{round_number}")
            if response.status_code == 200:
                break
            elif response.status_code in (404, 425):
                bt.logging.debug(f"Randomness for round {round_number} is not yet available.")
                if not retry_if_too_early:
                    try:
                        response.raise_for_status()
                    except Exception as e:
                        raise TooEarly() from e
            elif response.status_code == 500:
                bt.logging.debug(f'{response.status_code} {response} {response.headers} {response.text}')
            time.sleep(self.RETRY_BACKOFF_S)
            continue
        response.raise_for_status()
        bt.logging.debug(f"Got randomness for round {round_number} successfully.")
        
        return response.json()


class AbstractBittensorDrandTimelock:
    """Class for Drand-based timelock encryption and decryption using the timelock library."""
    #DRAND_URL: str = "https://api.drand.sh"  # more 500 than 200
    DRAND_URL: str = "https://drand.cloudflare.com"

    def __init__(self) -> None:
        """Initialize the Timelock client."""
        self.tl = timelock.Timelock(self.PK_HEX)
        self.drand_client = DrandClient(f'{self.DRAND_URL}/{self.CHAIN}')

    def _get_drand_round_info(self, round_number: int, cache: Dict[int, str]):
        """Fetch the randomness for a given round, using a cache to prevent duplicate requests."""
        if not (round_info := cache.get(round_number)):
            try:
                round_info = cache[round_number] = self.drand_client.get(round_number)
            except ValueError:
                raise RuntimeError(f"Randomness for round {round_number} is not yet available.")
        return round_info

    def _get_drand_signature(self, round_number: int, cache: Dict[int, str]) -> str:
        return bytearray.fromhex(
            self._get_drand_round_info(round_number, cache)['signature']
        )

    def get_current_round(self) -> int:
        return int(time.time()- self.NET_START) // self.ROUND_DURATION 

    def encrypt(self, uid: int, message: str, rounds: int = 1) -> Tuple[int, bytes]:
        """
        Encrypt a message with a future Drand round key, prefixing it with the UID.
        Returns a tuple of (target_round, encrypted_message).
        """
        target_round: int = self.get_current_round() + rounds
        bt.logging.info(f"Encrypting message for UID {uid}... Unlockable at round {target_round}")

        prefixed_message: str = f"{uid}:{message}"
        sk = secrets.token_bytes(32)  # an ephemeral secret key
        ciphertext: bytes = self.tl.tle(target_round, prefixed_message, sk)

        return target_round, ciphertext

    def decrypt(self, uid: int, ciphertext: bytes, target_round: int, signature: Optional[str] = None) -> Optional[str]:
        """
        Attempt to decrypt a single message, verifying the UID prefix.
        If the decrypted message doesn't start with the expected UID prefix, return None.
        """
        if not signature:
            try:
                signature: bytes = self._get_drand_signature(target_round, {})
            except RuntimeError as e:
                bt.logging.error(e)
                raise

        bt.logging.info(f"Decrypting message for UID {uid} at round {target_round}...")

        # key: bytes = self._derive_key(randomness)
        # cipher: Fernet = Fernet(key)
        # decrypted_message: str = cipher.decrypt(encrypted_message).decode()
        print(repr(ciphertext))
        try:
            plaintext = self.tl.tld(ciphertext, signature).decode()
        except Exception as e:
            bt.logging.error(f"Error decrypting message: {e}")
            return None

        expected_prefix = f"{uid}:"
        if not plaintext.startswith(expected_prefix):
            bt.logging.warning(f"UID mismatch: Expected {expected_prefix} but got {plaintext}")
            return None

        return plaintext[len(expected_prefix):]

    def decrypt_dict(self, encrypted_dict: Dict[int, Tuple[int, bytes]]) -> Dict[int, Optional[str]]:
        """
        Decrypt a dictionary of {uid: (target_round, encrypted_payload)}, caching signatures for this function call.
        """
        decrypted_dict: Dict[int, Optional[bytes]] = {}
        cache: Dict[int, str] = {}

        for uid, (target_round, ciphertext) in encrypted_dict.items():
            try:
                signature = self._get_drand_signature(target_round, cache)
                decrypted_dict[uid] = self.decrypt(uid, ciphertext, target_round, signature)
            except RuntimeError:
                current_round = self.get_current_round()
                bt.logging.warning(f"Skipping UID {uid}: Too early to decrypt: {target_round=}, {current_round=}")
                decrypted_dict[uid] = None
                continue
            except ValueError:
                bt.logging.warning(f"Skipping UID {uid}: Invalid ciphertext")
                decrypted_dict[uid] = None
                continue
            #print(repr(ciphertext))            
        return decrypted_dict


#class BittensorDrandTimelock(AbstractBittensorDrandTimelock):
#    ROUND_DURATION = 30
#    PK_HEX = '868f005eb8e6e4ca0a47c8a77ceaa5309a47978a7c71bc5cce96366b5d7a569937c529eeda66c7293784a9402801af31'
#    CHAIN = '8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce'
#    NET_START = 1595431050
 

class QuicknetBittensorDrandTimelock(AbstractBittensorDrandTimelock):
    ROUND_DURATION = 3
    PK_HEX = "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
    CHAIN = '52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971'
    NET_START = 1692803367


def _prepare_test(bdt):
    msg1: str = "Secret message #1"
    msg2: str = "Secret message #2"

    encrypted_dict: Dict[int, Tuple[int, bytes]] = {
        1: bdt.encrypt(1, msg1, rounds=1),
        2: bdt.encrypt(2, msg2, rounds=15),
    }
    bt.logging.info(f"Encrypted Dictionary: {encrypted_dict}")
    return encrypted_dict


def sync_decrypt_example(encrypted_dict, bdt) -> None:
    """Synchronous example of encryption and decryption."""
    try:
        decrypted_dict: Dict[int, Optional[str]] = bdt.decrypt_dict(encrypted_dict)
        logger.info(f"Decrypted Dictionary: {decrypted_dict}")
    except RuntimeError:
        logger.error("Decryption failed for one or more entries.")


async def async_decrypt_example(encrypted_dict, bdt) -> None:
    """Example of using BittensorDrandTimelock in async code via ThreadPoolExecutor."""
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        try:
            decrypted_dict = await loop.run_in_executor(executor, bdt.decrypt_dict, encrypted_dict)
            logger.info(f"Decrypted Dictionary: {decrypted_dict}")
        except RuntimeError:
            logger.error("Decryption failed for one or more entries.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    for bdt in (
            QuicknetBittensorDrandTimelock(),
            #BittensorDrandTimelock(),  # something wrong with the public key?
        ):
        encrypted_dict = _prepare_test(bdt)
        time.sleep(bdt.ROUND_DURATION + 1)
        print('='*50)
        sync_decrypt_example(encrypted_dict, bdt)
        print('='*50)
        asyncio.run(
            async_decrypt_example(
                encrypted_dict,
                bdt,
            )
        )
        print('#'*75)
