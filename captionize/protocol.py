# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import bittensor as bt
from typing import Optional, List

class OCRSynapse(bt.Synapse):
    """
    A simple OCR synapse protocol representation which uses bt.Synapse as its base.
    This protocol enables communication between the miner and the validator.

    Attributes:
    - base64_image: Base64 encoding of pdf image to be processed by the miner.
    - response: List[dict] containing data extracted from the image.
    """

    # Used by the validator
    time_elapsed = 0

    # Required request input, filled by sending dendrite caller. It is a base64 encoded string.
    base64_image: str

    # Optional request output, filled by recieving axon.
    response: Optional[List[dict]] = None

    def deserialize(self) -> List[dict]:
        """
        Deserialize the miner response.

        Returns:
        - List[dict]: The deserialized response, which is a list of dictionaries containing the extracted data.
        """
        return self.response
