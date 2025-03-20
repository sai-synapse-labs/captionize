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
import random
import captionize  # our project package
from captionize.base.validator import BaseValidatorNeuron
from captionize.validator.generate import generate_synthetic_jobs
from captionize.validator.reward import get_rewards
from captionize.utils.uids import get_random_uids
import torch

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
        jobs_data = generate_synthetic_jobs()
        bt.logging.debug(f"Generated job: {jobs_data}")
        
        # Select one job randomly.
        job = random.choice(jobs_data) if jobs_data else None
        if job is None:
            bt.logging.error("No job generated.")
            return
        
        if not isinstance(job, dict):
            bt.logging.error("Selected job is not a dictionary. Received: {}".format(type(job)))
            return

        # Create ground-truth labels (here as a list of one segment with dummy timing)
        labels = [
            { 
                "start_time": 0.0, 
                "end_time": 0.0, 
                "text": job.get("normalized_text", ""), 
                "gender": job.get("gender", "unknown")
            }
        ]
        
        # Create a CaptionSynapse with the job data.
        synapse = captionize.protocol.CaptionSynapse(
            job_id=job.get("job_id"),
            base64_audio=job.get("audio"),
            audio_path=job.get("audio_path"),
            language="en",
            miner_state="in_progress"
        )
        
        # Query miners with the synapse. Assume dendrite.query returns a list of responses.
        responses = await self.dendrite.forward(
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=synapse,
            deserialize=False,  # we want the raw response for further processing
        )
        
        # If no responses, handle gracefully
        if responses is None:
            responses = []
        
        bt.logging.info(f"Received responses: {responses}")
        
        # Compute rewards for the miner responses
        rewards = get_rewards(self, labels=labels, responses=responses)
        bt.logging.info(f"Scored responses: {rewards}")
        
        # Update miner scores using the computed rewards (update_scores must be implemented in BaseValidatorNeuron)
        self.update_scores(rewards, miner_uids)
        
        return synapse

    def set_weights(self):
        """
        Override the base validator's set_weights method to handle tensor conversions properly.
        Sets the validator weights for each miner UID based on their performance scores.
        """
        # Check if scores contain NaN values
        if torch.isnan(self.scores).any():
            bt.logging.warning("Scores contain NaN values. This may indicate an issue with reward calculations.")
            # Replace NaN with zeros to prevent errors
            self.scores = torch.nan_to_num(self.scores, 0.0)
        
        # Ensure scores is a PyTorch tensor on the correct device
        if not isinstance(self.scores, torch.Tensor):
            self.scores = torch.tensor(self.scores, dtype=torch.float32).to(self.device)
        
        # Normalize scores to get weights (sum to 1)
        weights = torch.nn.functional.normalize(self.scores, p=1, dim=0)
        bt.logging.debug(f"Normalized weights: {weights}")
        
        # Process weights for chain compatibility
        try:
            # Convert PyTorch tensor to NumPy array (this was missing before)
            weights_numpy = weights.detach().cpu().numpy()
            
            processed_weight_uids, processed_weights = bt.utils.weight_utils.process_weights_for_netuid(
                uids=self.metagraph.uids,
                weights=weights_numpy,  # Use the NumPy array here
                netuid=self.config.netuid,
                subtensor=self.subtensor,
                metagraph=self.metagraph,
            )
            
            bt.logging.debug(f"Processed weights: {processed_weights}")
            bt.logging.debug(f"Processed weight UIDs: {processed_weight_uids}")
            
            # Set weights on chain
            self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=processed_weight_uids,
                weights=processed_weights,
                wait_for_finalization=False,
                version_key=self.spec_version,
            )
            
            bt.logging.info(f"Set weights: {processed_weights}")
            
        except Exception as e:
            bt.logging.error(f"Failed to set weights: {e}")
            import traceback
            bt.logging.debug(traceback.format_exc())

if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(5)