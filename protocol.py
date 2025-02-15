import typing
import bittensor as bt


class ChallengeSynapse(bt.Synapse):
    """
    Attributes:
    - target_protein (str): The protein or query sent by the validator.
    - product_name (typing.Optional[str]): The miner's proposed molecule ID.
      This is optional because the validator only sets the target_protein,
      while the miner sets product_name in the response.
    """
    
    # Required request input, provided by the validator.
    target_protein: str

    # Optional response output, provided by the miner.
    product_name: typing.Optional[str] = None
