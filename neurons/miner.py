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

        # Determine device before loading models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bt.logging.info(f"Using device: {self.device}")

        # Load SpeechBrain ASR model for transcription.
        asr_source = "speechbrain/asr-crdnn-rnnlm-librispeech"  # or choose another suitable model
        self.asr_model = EncoderDecoderASR.from_hparams(source=asr_source, savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")
        self.asr_model.eval()
        
        # Load the pretrained voice gender classifier.
        self.gender_model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
        self.gender_model.eval()
        
        # Move models to the appropriate device
        try:
            self.asr_model.to(self.device)
            self.gender_model.to(self.device)
            bt.logging.info(f"Models moved to {self.device}")
        except Exception as e:
            bt.logging.error(f"Error moving models to device: {e}")
            if self.device.type == 'cuda':
                bt.logging.warning("Falling back to CPU")
                self.device = torch.device("cpu")
                self.asr_model.to(self.device)
                self.gender_model.to(self.device)

        bt.logging.info(f"Loaded ASR model from {asr_source} and voice gender classifier.")

    async def forward(self, synapse: CaptionSynapse) -> CaptionSynapse:
        """
        Process the incoming CaptionSynapse request.
        
        If the job has not been processed (i.e. job_status is not "done"),
        it:
          - Obtains the audio file path from synapse (or decodes the base64_audio to a temporary file).
          - Uses SpeechBrain to transcribe the audio.
          - Uses the voice gender classifier to predict gender.
          - Updates the synapse with a transcript (wrapped as a single segment) and predicted gender.
          - Updates job_status to "done" and records the time_elapsed.
        
        Args:
            synapse (CaptionSynapse): The synapse with incoming job data.
        
        Returns:
            CaptionSynapse: The synapse with attached transcription results and gender prediction.
        """
        start_time = time.time()
        # Process only if job_status is not "done"
        if synapse.job_status not in ["done"]:
            # Use audio_path if provided; otherwise, decode base64_audio and write to a temporary file.
            if synapse.audio_path and os.path.exists(synapse.audio_path):
                audio_file = synapse.audio_path
            else:
                temp_filename = f"/tmp/{uuid.uuid4()}.wav"
                audio_bytes = base64.b64decode(synapse.base64_audio)
                with open(temp_filename, "wb") as f:
                    f.write(audio_bytes)
                audio_file = temp_filename

            try:
                # Transcribe the audio using SpeechBrain ASR model
                bt.logging.debug(f"Transcribing file: {audio_file}")
                transcript = self.asr_model.transcribe_file(audio_file)
                
                # Gender classification can have device issues, handle carefully
                bt.logging.debug(f"Predicting gender from file: {audio_file}")
                try:
                    # Ensure model is in eval mode and on the right device
                    self.gender_model.eval()
                    self.gender_model.to(self.device)
                    
                    # Make sure the ECAPA model's internal device setting matches too
                    if hasattr(self.gender_model, 'device'):
                        self.gender_model.device = self.device
                    
                    # Option 1: Use the model's predict method with explicit device parameter
                    predicted_gender = self.gender_model.predict(audio_file, device=self.device)
                    
                except Exception as e:
                    bt.logging.error(f"Error in gender prediction: {e}")
                    bt.logging.info("Attempting fallback gender prediction on CPU")
                    
                    # Option 2: Move model to CPU for prediction as a fallback
                    try:
                        self.gender_model.to('cpu')
                        if hasattr(self.gender_model, 'device'):
                            self.gender_model.device = torch.device('cpu')
                        predicted_gender = self.gender_model.predict(audio_file, device='cpu')
                        # Move back to original device after prediction
                        self.gender_model.to(self.device)
                    except Exception as e2:
                        bt.logging.error(f"Fallback gender prediction also failed: {e2}")
                        predicted_gender = "unknown"
                
            except Exception as e:
                bt.logging.error(f"Error during transcription/gender recognition: {e}")
                synapse.job_status = "failed"
                transcript = ""
                predicted_gender = None

            # If we created a temporary file, remove it.
            if (not synapse.audio_path) and os.path.exists(audio_file):
                os.remove(audio_file)

            # Update synapse with transcription result and predicted gender
            synapse.segments = [{"text": transcript}]
            synapse.predicted_gender = predicted_gender  # Store in the dedicated field
            synapse.time_elapsed = time.time() - start_time  # Now a regular field
            synapse.job_status = "done"
        else:
            bt.logging.info("Job already processed; skipping transcription.")
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