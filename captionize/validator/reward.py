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

import torch
import bittensor as bt
from typing import List
import editdistance
from scipy.optimize import linear_sum_assignment

from captionize.protocol import CaptionSynapse  # our custom synapse for captioning

def get_text_reward(text1: str, text2: str = None) -> float:
    """
    Calculate a normalized text reward based on edit distance.
    A reward of 1.0 means a perfect match.

    Args:
      text1 (str): The reference transcript.
      text2 (str): The predicted transcript.

    Returns:
      float: Normalized text reward between 0 and 1.
    """
    if not text2:
        return 0.0
    return 1 - editdistance.eval(text1, text2) / max(len(text1), len(text2))

def get_gender_reward(gender_true: str, gender_pred: str = None) -> float:
    """
    Compute a reward based on the comparison of speaker gender.
    
    Returns 1.0 if the genders match (case-insensitive), otherwise 0.0.
    
    Args:
      gender_true (str): Ground truth speaker gender.
      gender_pred (str, optional): Predicted speaker gender.
      
    Returns:
      float: 1.0 if genders match, 0.0 otherwise.
    """
    if not gender_pred:
        return 0.0
    return 1.0 if gender_true.strip().lower() == gender_pred.strip().lower() else 0.0

def section_reward(label: dict, pred: dict, alpha_t: float = 1.0, verbose: bool = False) -> dict:
    """
    Compute a reward for a single caption segment based on text quality.
    
    Args:
      label (dict): Ground truth segment with "text".
      pred (dict): Predicted segment with "text".
      alpha_t (float): Weight factor for text reward.
      verbose (bool): If True, logs additional details.
      
    Returns:
      dict: Contains the text reward and a total reward (scaled by alpha_t).
    """
    text_reward = get_text_reward(label['text'], pred.get('text'))
    
    # Ensure alpha_t is a valid number
    if alpha_t is None:
        bt.logging.warning("alpha_t was None in section_reward, using default value 1.0")
        alpha_t = 1.0
    
    total = alpha_t * text_reward
    if verbose:
        bt.logging.info(', '.join([f"text_reward: {text_reward:.3f}", f"total: {total:.3f}"]))
    return {"text": text_reward, "total": total}

def sort_predictions(labels: List[dict], predictions: List[dict]) -> List[dict]:
    """
    Align the predicted segments with the ground truth segments using the Hungarian algorithm.
    
    Args:
      labels (List[dict]): List of ground truth segments.
      predictions (List[dict]): List of predicted segments.
      
    Returns:
      List[dict]: Sorted predictions aligned with labels.
    """
    # Pad predictions if needed
    predictions += [{}] * (len(labels) - len(predictions))
    
    # Create reward matrix
    r = torch.zeros((len(labels), len(predictions)))
    for i in range(len(labels)):
        for j in range(len(predictions)):
            r[i, j] = section_reward(labels[i], predictions[j])["total"]
    
    # Convert to NumPy for scipy, then back to native Python types
    row_ind, col_ind = linear_sum_assignment(r.detach().cpu().numpy(), maximize=True)
    col_ind = col_ind.tolist()  # Convert NumPy array to Python list
    
    # Sort the predictions based on the assignment
    sorted_preds = [predictions[i] for i in col_ind]
    return sorted_preds

def reward(self, labels: List[dict], response: CaptionSynapse) -> float:
    """
    Compute the overall reward for a miner's response to a caption task.
    The reward combines text accuracy, response time, and gender accuracy.
    
    Args:
      labels (List[dict]): Ground truth segments, each with "text" and "gender" fields.
      response (CaptionSynapse): The miner's response, expected to include:
           - segments: a list of predicted CaptionSegment objects (each with "text")
           - time_elapsed: time taken by the miner to produce the response
           - predicted_gender (optional): predicted speaker gender.
    
    Returns:
      float: The final combined reward.
    """
    # Get segments from response
    predictions = response.segments
    if predictions is None:
        return 0.0

    # Check segment type and convert if needed
    # If segments are already dictionaries, use them directly
    # Otherwise, convert them to dictionaries using .dict() method
    segment_dicts = []
    for seg in predictions:
        if isinstance(seg, dict):
            segment_dicts.append(seg)  # Already a dict, use as is
        else:
            try:
                segment_dicts.append(seg.dict())  # Try to convert to dict if it has a dict() method
            except AttributeError:
                bt.logging.warning(f"Segment is not a dict and has no .dict() method: {seg}")
                # Create a basic dict with just the text if possible
                if hasattr(seg, 'text'):
                    segment_dicts.append({"text": seg.text})
                else:
                    segment_dicts.append({})  # Empty dict as fallback

    # Align predictions with ground truth segments
    aligned_predictions = sort_predictions(labels, segment_dicts)
    
    # Retrieve weight factors from configuration with defaults
    try:
        alpha_text = float(getattr(self.config.neuron, 'alpha_text', 1.0))
        if alpha_text is None:
            alpha_text = 1.0
    except (ValueError, TypeError):
        bt.logging.warning("Invalid alpha_text in config, using default 1.0")
        alpha_text = 1.0
    
    try:
        alpha_prediction = float(getattr(self.config.neuron, 'alpha_prediction', 0.7))
        if alpha_prediction is None or alpha_prediction <= 0:
            alpha_prediction = 0.7
    except (ValueError, TypeError):
        bt.logging.warning("Invalid alpha_prediction in config, using default 0.7")
        alpha_prediction = 0.7
    
    try:
        alpha_time = float(getattr(self.config.neuron, 'alpha_time', 0.2))
        if alpha_time is None or alpha_time < 0:
            alpha_time = 0.2
    except (ValueError, TypeError):
        bt.logging.warning("Invalid alpha_time in config, using default 0.2")
        alpha_time = 0.2
    
    try:
        alpha_gender = float(getattr(self.config.neuron, 'alpha_gender', 0.1))
        if alpha_gender is None or alpha_gender < 0:
            alpha_gender = 0.1
    except (ValueError, TypeError):
        bt.logging.warning("Invalid alpha_gender in config, using default 0.1")
        alpha_gender = 0.1
    
    try:
        timeout = float(getattr(self.config.neuron, 'timeout', 10.0))
        if timeout is None or timeout <= 0:
            timeout = 10.0
    except (ValueError, TypeError):
        bt.logging.warning("Invalid timeout in config, using default 10.0")
        timeout = 10.0

    # Compute text reward from each segment
    section_rewards = [
        section_reward(label, pred, alpha_t=alpha_text, verbose=True)
        for label, pred in zip(labels, aligned_predictions)
    ]
    text_reward_values = [r["total"] for r in section_rewards]
    prediction_reward = torch.mean(torch.FloatTensor(text_reward_values))
    
    # Compute time reward
    time_reward = max(1 - response.time_elapsed / timeout, 0)
    
    # Compute gender reward using predicted_gender field
    gender_true = labels[0].get("gender", "").strip()
    gender_pred = getattr(response, "predicted_gender", None)
    gender_reward = get_gender_reward(gender_true, gender_pred)
    
    # Combine rewards using weighted average
    # weight_sum = alpha_prediction + alpha_time + alpha_gender
    weight_sum = alpha_prediction + alpha_gender
    if weight_sum <= 0:
        bt.logging.warning("Sum of weights is zero or negative, using equal weights")
        alpha_prediction = alpha_gender = 1.0
        weight_sum = 2.0
    
    total_reward = (alpha_prediction * prediction_reward + alpha_gender * gender_reward) / weight_sum
    
    bt.logging.info(f"Prediction Reward: {prediction_reward:.3f}, Gender Reward: {gender_reward:.3f}, Total Reward: {total_reward:.3f}")
    return total_reward

def get_rewards(self, labels: List[dict], responses: List[CaptionSynapse]) -> torch.FloatTensor:
    """
    Compute rewards for a batch of miner responses.
    
    Args:
      labels (List[dict]): The ground truth segments.
      responses (List[CaptionSynapse]): List of miner responses.
    
    Returns:
      torch.FloatTensor: Tensor of computed rewards.
      
    Note:
      If responses is None, returns a zero tensor of shape (1,).
      This handles the case where dendrite.query() returns None.
    """
    # Handle empty responses or None
    if responses is None or len(responses) == 0:
        return torch.zeros(1).to(self.device)
    
    # Calculate rewards for each response
    rewards = []
    for response in responses:
        if response is None:
            rewards.append(0.0)  # Add zero reward for None response
        else:
            rewards.append(reward(self, labels, response))
    
    # Convert to PyTorch tensor and return
    return torch.FloatTensor(rewards).to(self.device)