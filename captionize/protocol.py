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
from typing import Optional, List, ClassVar, Dict


class CaptionSynapse(bt.Synapse):
    """
    A caption synapse protocol for the Captionise subnet.
    
    This synapse enables communication between miners and validators by carrying:
      - job_id: A unique identifier for the captioning task.
      - base64_audio: The audio data (encoded in base64) to be transcribed.
      - language: The language code for transcription (default "en").
      - segments: A list of dictionaries representing transcribed segments. Each dictionary should contain:
            - "start_time": (float) the start time of the segment in seconds,
            - "end_time": (float) the end time of the segment in seconds,
            - "text": (str) the transcribed text.
            - "gender": (str) the gender of the speaker.      
      - job_status: Optional status information from the miner.
      - time_elapsed: Time taken to process the request in seconds.
    """
    job_id: Optional[str] = None
    base64_audio: Optional[str] = None
    audio_path: Optional[str] = None
    language: Optional[str] = "en"
    segments: Optional[List[dict]] = None    # Each dict: {"start_time": float, "end_time": float, "text": str, "gender": str}
    job_status: Optional[str] = None
    time_elapsed: Optional[float] = 0.0      # Changed from ClassVar to regular field
    predicted_gender: Optional[str] = None    # Added for storing gender prediction

    def deserialize(self) -> "CaptionSynapse":
        """
        Deserialize the miner response.
        
        This method can be extended to perform additional post-processing
        on the segments if necessary. Here, it simply logs and returns self.
        
        Returns:
            CaptionSynapse: The deserialized synapse instance.
        """
        bt.logging.info(f"Deserializing CaptionSynapse for job_id: {self.job_id}")
        if self.segments is not None:
            bt.logging.debug(f"Segments: {self.segments}")
        return self