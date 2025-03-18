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
import uuid
import base64
import asyncio
import typing
import torch
import bittensor as bt


# Import our protocol and base miner class
from captionize.protocol import CaptionSynapse
from captionize.base.miner import BaseMinerNeuron

# Import SpeechBrain ASR model and the voice gender classifier from Hugging Face
from speechbrain.inference.ASR import EncoderDecoderASR
from captionize.model.base_GRM import ECAPA_gender

class Miner(BaseMinerNeuron):
    """
    Captionise miner neuron class.
    This miner uses SpeechBrain for transcription and a pretrained voice gender classifier
    to predict speaker gender. It processes incoming CaptionSynapse requests and returns a synapse
    containing the transcript, predicted gender, and processing time.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        # Determine device - prefer GPU for performance
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            bt.logging.info(f"Using device: {self.device}")
        except:
            self.device = torch.device("cpu")
            bt.logging.warning("Error setting device, falling back to CPU")

        # Load models with careful error handling
        try:
            # Load SpeechBrain ASR model for transcription
            asr_source = "speechbrain/asr-crdnn-rnnlm-librispeech"
            self.asr_model = EncoderDecoderASR.from_hparams(source=asr_source, savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")
            self.asr_model.eval()
            
            # Load the pretrained voice gender classifier
            self.gender_model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
            self.gender_model.eval()
            
            # Try to move models to GPU (with fallback to CPU)
            if self.device.type == 'cuda':
                try:
                    self.asr_model.to(self.device)
                    self.gender_model.to(self.device)
                    bt.logging.info(f"Models moved to {self.device}")
                except Exception as e:
                    bt.logging.warning(f"Failed to move models to {self.device}: {e}")
                    self.device = torch.device("cpu")
                    bt.logging.info("Falling back to CPU")
            
            bt.logging.info(f"Loaded ASR model from {asr_source} and voice gender classifier on {self.device}")
        except Exception as e:
            bt.logging.error(f"Error initializing models: {e}")
            import traceback
            bt.logging.debug(traceback.format_exc())
            raise

    async def forward(self, synapse: CaptionSynapse) -> CaptionSynapse:
        """
        Process the incoming CaptionSynapse request with GPU acceleration and fallbacks.
        """
        start_time = time.time()
        if synapse.job_status not in ["done"]:
            # Get or create audio file
            if synapse.audio_path and os.path.exists(synapse.audio_path):
                audio_file = synapse.audio_path
            else:
                temp_filename = f"/tmp/{uuid.uuid4()}.wav"
                audio_bytes = base64.b64decode(synapse.base64_audio)
                with open(temp_filename, "wb") as f:
                    f.write(audio_bytes)
                audio_file = temp_filename

            transcript = ""
            predicted_gender = "unknown"
            used_device = self.device  # Start with configured device
            
            # Process with error handling and GPU fallback
            try:
                # Try transcription with current device setting
                bt.logging.debug(f"Transcribing file using {used_device}: {audio_file}")
                
                # Try using GPU first
                if used_device.type == 'cuda':
                    try:
                        # Ensure model is on the right device
                        self.asr_model.to(used_device)
                        transcript = self.asr_model.transcribe_file(audio_file)
                        bt.logging.debug(f"Transcription result (GPU): '{transcript}'")
                    except Exception as gpu_err:
                        # Fall back to CPU if GPU fails
                        bt.logging.warning(f"GPU transcription failed, falling back to CPU: {gpu_err}")
                        used_device = torch.device("cpu")
                        self.asr_model.to(used_device)
                        transcript = self.asr_model.transcribe_file(audio_file)
                        bt.logging.debug(f"Transcription result (CPU fallback): '{transcript}'")
                else:
                    # Use CPU directly if that's our primary device
                    transcript = self.asr_model.transcribe_file(audio_file)
                    
                # Try gender prediction on same device as successful transcription
                try:
                    bt.logging.debug(f"Predicting gender on {used_device} from file: {audio_file}")
                    # Move gender model to same device as successful transcription
                    self.gender_model.to(used_device)
                    if hasattr(self.gender_model, 'device'):
                        # Some models have internal device tracking
                        self.gender_model.device = used_device
                    predicted_gender = self.gender_model.predict(audio_file, device=used_device)
                    bt.logging.debug(f"Gender prediction result: {predicted_gender}")
                except Exception as e:
                    # Fall back to CPU for gender prediction if needed
                    bt.logging.warning(f"Gender prediction on {used_device} failed: {e}")
                    if used_device.type == 'cuda':
                        bt.logging.info("Trying gender prediction on CPU instead")
                        try:
                            self.gender_model.to('cpu')
                            if hasattr(self.gender_model, 'device'):
                                self.gender_model.device = torch.device('cpu')
                            predicted_gender = self.gender_model.predict(audio_file, device='cpu')
                            bt.logging.debug(f"Gender prediction result (CPU fallback): {predicted_gender}")
                        except Exception as e2:
                            bt.logging.error(f"CPU fallback gender prediction also failed: {e2}")
                            predicted_gender = "unknown"
                    else:
                        predicted_gender = "unknown"
                    
            except Exception as e:
                bt.logging.error(f"Error during processing: {e}")
                import traceback
                bt.logging.debug(traceback.format_exc())
                synapse.job_status = "failed"

            # Cleanup temp file
            if (not synapse.audio_path) and os.path.exists(audio_file):
                os.remove(audio_file)

            # Update synapse with results
            synapse.segments = [{"text": transcript}]
            synapse.predicted_gender = predicted_gender
            synapse.time_elapsed = time.time() - start_time
            synapse.job_status = "done"
            
        return synapse

    async def blacklist(self, synapse: CaptionSynapse) -> typing.Tuple[bool, str]:
        """
        Determine if a request should be blacklisted.
        For now, this simple logic blacklists unrecognized hotkeys.
        """
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            bt.logging.trace(f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}")
            return True, "Unrecognized hotkey"
        bt.logging.trace(f"Hotkey recognized: {synapse.dendrite.hotkey}")
        return False, "Hotkey recognized"

    async def priority(self, synapse: CaptionSynapse) -> float:
        """
        Determine the priority for an incoming request based on the caller's stake.
        """
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(self.metagraph.S[caller_uid])
        bt.logging.trace(f"Priority for {synapse.dendrite.hotkey}: {priority}")
        return priority

if __name__ == "__main__":
    # Running the miner in standalone mode for testing.
    with Miner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)