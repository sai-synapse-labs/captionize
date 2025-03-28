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

import os
import time
import hashlib
import bittensor as bt

import captionize  # our project package
from captionize.base.validator import BaseValidatorNeuron
from captionize.validator.generate import generate_synthetic_job
from captionize.validator.reward import get_rewards
from captionize.utils.uids import get_random_uids

class Validator(BaseValidatorNeuron):
    """
    Captionise validator neuron class.
    
    This class inherits from BaseValidatorNeuron, which handles wallet, subtensor, metagraph,
    logging, and configuration. The Validator:
      - Generates a synthetic caption job from the VoxPopuli dataset.
      - Dispatches the task to miners via a CaptionSynapse.
      - Collects miner responses and scores them.
      - Updates miner scores accordingly.
    """
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        bt.logging.info("Loading validator state...")
        self.load_state()
        
        # Define a directory for storing temporary data (if needed)
        self.data_dir = './data/audio/'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    async def forward(self):
        """
        Called periodically by the validator.
        
        Steps:
          1. Retrieve random miner UIDs.
          2. Generate a synthetic caption job using VoxPopuli.
          3. Create a CaptionSynapse with the generated job data.
          4. Query miners with this synapse.
          5. Score the responses.
          6. Update miner scores.
        """
        # Get random miner UIDs from the metagraph (helper function)
        # miner_uids = get_random_uids(self, k=min(self.config.neuron.sample_size, self.metagraph.n.item()))
        miner_uids = get_random_uids(self, k=min(self.config.neuron.sample_size, len(self.metagraph.uids)))
        bt.logging.debug(f"Miner UIDs: {miner_uids}")
        
        # Generate a synthetic caption job from VoxPopuli
        job_data = generate_synthetic_job()
        bt.logging.debug(f"Generated job: {job_data}")
        
        # For scoring, create ground-truth labels using the transcript.
        # Here, we wrap the transcript in a list of one segment with a default gender.
        labels = [{"start_time": 0.0, "end_time": 0.0, "text": job_data["transcript"], "gender": "unknown"}]
        
        # Create a CaptionSynapse with the job data.
        synapse = captionize.protocol.CaptionSynapse(
            job_id=job_data["job_id"],
            base64_audio=job_data["base64_audio"],
            audio_path=job_data["audio_path"],
            language="en"
        )
        
        # Query miners with the synapse. Assume dendrite.query returns a list of responses.
        responses = self.dendrite.query(
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=synapse,
            deserialize=False,  # we want the raw response for further processing
        )
        
        bt.logging.info(f"Received responses: {responses}")
        
        # Compute rewards for the miner responses
        rewards = get_rewards(self, labels=labels, responses=responses)
        bt.logging.info(f"Scored responses: {rewards}")
        
        # Update miner scores using the computed rewards (update_scores must be implemented in BaseValidatorNeuron)
        self.update_scores(rewards, miner_uids)

if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(5)