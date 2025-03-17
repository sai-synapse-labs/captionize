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
    predictions += [{}] * (len(labels) - len(predictions))
    r = torch.zeros((len(labels), len(predictions)))
    for i in range(len(labels)):
        for j in range(len(predictions)):
            r[i, j] = section_reward(labels[i], predictions[j])["total"]
    row_ind, col_ind = linear_sum_assignment(r.numpy(), maximize=True)
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
    predictions = response.segments
    if predictions is None:
        return 0.0

    # Align predictions with ground truth segments
    predictions = sort_predictions(labels, [seg.dict() for seg in predictions])
    
    # Retrieve weight factors from configuration
    alpha_text = self.config.neuron.alpha_text    # e.g., weight for text quality
    alpha_prediction = self.config.neuron.alpha_prediction  # overall prediction weight
    alpha_time = self.config.neuron.alpha_time      # weight for response time
    alpha_gender = self.config.neuron.alpha_gender  # weight for gender matching

    # Compute text reward from each segment
    section_rewards = [
        section_reward(label, pred, alpha_t=alpha_text, verbose=True)
        for label, pred in zip(labels, predictions)
    ]
    text_reward_values = [r["total"] for r in section_rewards]
    prediction_reward = torch.mean(torch.FloatTensor(text_reward_values))
    
    # Compute time reward: assuming response has attribute time_elapsed
    time_reward = max(1 - response.time_elapsed / self.config.neuron.timeout, 0)
    
    # Compute gender reward: assume ground truth is in labels[0]["gender"] and miner provides predicted_gender
    gender_true = labels[0].get("gender", "").strip()
    gender_pred = getattr(response, "predicted_gender", None)
    gender_reward = get_gender_reward(gender_true, gender_pred)
    
    # Combine rewards using weighted average
    total_reward = (alpha_prediction * prediction_reward + alpha_time * time_reward + alpha_gender * gender_reward) \
                   / (alpha_prediction + alpha_time + alpha_gender)
    
    bt.logging.info(f"Prediction Reward: {prediction_reward:.3f}, Time Reward: {time_reward:.3f}, Gender Reward: {gender_reward:.3f}, Total Reward: {total_reward:.3f}")
    return total_reward

def get_rewards(self, labels: List[dict], responses: List[CaptionSynapse]) -> torch.FloatTensor:
    """
    Compute rewards for a batch of miner responses.
    
    Args:
      labels (List[dict]): The ground truth segments.
      responses (List[CaptionSynapse]): List of miner responses.
    
    Returns:
      torch.FloatTensor: Tensor of computed rewards.
    """
    rewards = [reward(self, labels, response) for response in responses]
    return torch.FloatTensor(rewards).to(self.device)