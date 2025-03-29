# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import bittensor as bt
from typing import List, Dict, Any
import editdistance
import re
import string
from scipy.optimize import linear_sum_assignment
import jiwer

from captionize.protocol import CaptionSynapse  # our custom synapse for captioning

def get_text_reward(text1: str, text2: str = None) -> float:
    """
    Calculate a normalized text reward based on edit distance.
    A reward of 1.0 means a perfect match.
    Handles case-insensitive comparison for SpeechBrain ASR output.

    Args:
      text1 (str): The reference transcript.
      text2 (str): The predicted transcript.

    Returns:
      float: Normalized text reward between 0 and 1.
    """
    if not text2:
        return 0.0
    
    # Normalize whitespace
    text1 = ' '.join(text1.split())
    text2 = ' '.join(text2.split())
    
    return 1 - editdistance.eval(text1, text2) / max(len(text1), len(text2))

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis.
    
    Args:
        reference (str): The ground truth text
        hypothesis (str): The predicted text
        
    Returns:
        float: WER score (lower is better)
    """
    # Use jiwer to calculate WER
    try:
        return jiwer.wer(reference, hypothesis)
    except Exception as e:
        bt.logging.warning(f"Error calculating WER: {e}")
        return 1.0  # Return worst score on error

def get_wer_reward(reference: str, hypothesis: str) -> float:
    """
    Calculate a reward based on Word Error Rate (WER).
    A reward of 1.0 means a perfect match (WER = 0).
    
    Args:
        reference (str): The ground truth text
        hypothesis (str): The predicted text
        
    Returns:
        float: WER-based reward between 0 and 1
    """
    if not hypothesis or not reference:
        return 0.0
    
    wer = calculate_wer(reference, hypothesis)
    # Convert WER to reward (1 - WER, capped at 0)
    return max(0.0, 1.0 - wer)

def check_critical_words(reference: str, hypothesis: str) -> float:
    """
    Check if critical words in the reference are present in the hypothesis.
    Critical words are typically proper nouns, acronyms, or words in ALL CAPS.
    Modified to handle SpeechBrain's uppercase output format.
    
    Args:
        reference (str): The ground truth text
        hypothesis (str): The predicted text
        
    Returns:
        float: Reward between 0 and 1 based on critical word accuracy
    """
    if not hypothesis or not reference:
        return 0.0
    
    # For SpeechBrain output (which is often all caps),
    # we need a different approach to identify critical words
    
    # Split into words and remove punctuation
    ref_words = re.findall(r'\b\w+\b', reference)
    hyp_words = re.findall(r'\b\w+\b', hypothesis)
    
    # Consider proper nouns (words that start with capital in reference)
    # and acronyms (words with 2+ consecutive capitals)
    critical_words = []
    
    # If reference is not all uppercase, we can detect proper nouns
    if not reference.isupper():
        for word in ref_words:
            # Check if it's a proper noun (starts with capital) or acronym
            if (len(word) > 1 and word[0].isupper() and not word[1].isupper()) or \
               re.search(r'[A-Z]{2,}', word):
                critical_words.append(word.upper())
    else:
        # If reference is all uppercase (like SpeechBrain output),
        # consider words with 4+ characters as potentially important
        for word in ref_words:
            if len(word) >= 4:
                critical_words.append(word.upper())
    
    # If no critical words identified, return perfect score
    if not critical_words:
        return 1.0
    
    # Convert hypothesis words to uppercase for comparison
    hyp_words_upper = [word.upper() for word in hyp_words]
    
    # Count how many critical words are in the hypothesis
    correct_count = 0
    for word in critical_words:
        if word in hyp_words_upper:
            correct_count += 1
    
    return correct_count / len(critical_words)

def check_punctuation(reference: str, hypothesis: str) -> float:
    """
    Evaluate punctuation accuracy in the hypothesis compared to reference.
    
    Args:
        reference (str): The ground truth text
        hypothesis (str): The predicted text
        
    Returns:
        float: Reward between 0 and 1 based on punctuation accuracy
    """
    if not hypothesis or not reference:
        return 0.0
    
    # Extract punctuation from both texts
    ref_punct = [c for c in reference if c in string.punctuation]
    hyp_punct = [c for c in hypothesis if c in string.punctuation]
    
    # If no punctuation in reference, return perfect score
    if not ref_punct:
        return 1.0
    
    # Calculate punctuation precision and recall
    common_punct = set(ref_punct).intersection(set(hyp_punct))
    precision = len(common_punct) / max(1, len(hyp_punct))
    recall = len(common_punct) / len(ref_punct)
    
    # F1 score for punctuation
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def check_spelling(reference: str, hypothesis: str) -> float:
    """
    Evaluate spelling accuracy by comparing words in hypothesis to reference.
    
    Args:
        reference (str): The ground truth text
        hypothesis (str): The predicted text
        
    Returns:
        float: Reward between 0 and 1 based on spelling accuracy
    """
    if not hypothesis or not reference:
        return 0.0
    
    # Tokenize into words (remove punctuation first)
    ref_text = ''.join([c if c not in string.punctuation else ' ' for c in reference.lower()])
    hyp_text = ''.join([c if c not in string.punctuation else ' ' for c in hypothesis.lower()])
    
    ref_words = [w for w in ref_text.split() if w]
    hyp_words = [w for w in hyp_text.split() if w]
    
    if not ref_words:
        return 1.0
    
    # For each word in reference, find closest match in hypothesis
    total_similarity = 0.0
    for ref_word in ref_words:
        best_similarity = 0.0
        for hyp_word in hyp_words:
            # Calculate word similarity using character-level edit distance
            similarity = 1 - editdistance.eval(ref_word, hyp_word) / max(len(ref_word), len(hyp_word))
            best_similarity = max(best_similarity, similarity)
        total_similarity += best_similarity
    
    return total_similarity / len(ref_words)

def get_gender_reward(true_gender: str, pred_gender: str) -> float:
    """
    Calculate gender prediction reward.
    
    Args:
        true_gender (str): The ground truth gender
        pred_gender (str): The predicted gender
        
    Returns:
        float: 1.0 if correct, 0.0 if incorrect
    """
    if not pred_gender or not true_gender:
        return 0.0
    
    # Normalize gender strings for comparison
    true_norm = true_gender.strip().lower()
    pred_norm = pred_gender.strip().lower()
    
    # Check for exact match
    if true_norm == pred_norm:
        return 1.0
    
    # Check for partial matches (male/female vs man/woman)
    if (true_norm in ['male', 'man'] and pred_norm in ['male', 'man']) or \
       (true_norm in ['female', 'woman'] and pred_norm in ['female', 'woman']):
        return 1.0
    
    return 0.0

def sort_predictions(labels: List[dict], predictions: List[dict]) -> List[dict]:
    """
    Align predictions with ground truth segments using the Hungarian algorithm.
    
    Args:
        labels (List[dict]): Ground truth segments
        predictions (List[dict]): Predicted segments
        
    Returns:
        List[dict]: Aligned predictions
    """
    if not predictions:
        return []
    
    if not labels:
        return predictions
    
    # If we have only one label and one prediction, no need for alignment
    if len(labels) == 1 and len(predictions) == 1:
        return predictions
    
    # Create cost matrix for Hungarian algorithm
    cost_matrix = []
    for label in labels:
        row = []
        label_text = label.get("text", "")
        for pred in predictions:
            pred_text = pred.get("text", "")
            # Cost is 1 - similarity (higher similarity = lower cost)
            similarity = get_text_reward(label_text, pred_text)
            row.append(1.0 - similarity)
        cost_matrix.append(row)
    
    # Use Hungarian algorithm to find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create aligned predictions
    aligned_preds = []
    for i in range(len(labels)):
        if i < len(row_ind) and col_ind[i] < len(predictions):
            aligned_preds.append(predictions[col_ind[i]])
        else:
            # If no prediction is aligned with this label, add empty prediction
            aligned_preds.append({"text": ""})
    
    return aligned_preds

def get_rewards(self, labels: List[dict], responses: List[CaptionSynapse]) -> torch.FloatTensor:
    """
    Calculate rewards for multiple miner responses.
    
    Args:
        labels (List[dict]): Ground truth segments
        responses (List[CaptionSynapse]): Miner responses
        
    Returns:
        torch.FloatTensor: Tensor of rewards
    """
    rewards = []
    
    for response in responses:
        # Deserialize the response if needed
        if hasattr(response, 'deserialize'):
            response = response.deserialize()
        
        # Calculate reward for this response
        reward_value = reward(self, labels, response)
        rewards.append(reward_value)
    
    # Convert to tensor
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    return rewards_tensor

def reward(self, labels: List[dict], response: CaptionSynapse) -> float:
    """
    Compute the overall reward for a miner's response to a caption task.
    Optimized for SpeechBrain EncoderDecoderASR output format which typically
    produces uppercase text with minimal punctuation.
    
    Args:
      labels (List[dict]): Ground truth segments, each with "text" and "gender" fields.
      response (CaptionSynapse): The miner's response.
    
    Returns:
      float: The final combined reward.
    """
    # Get segments from response
    predictions = response.segments
    if predictions is None:
        return 0.0

    # Check segment type and convert if needed
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
    
    # Calculate rewards for each aspect of the transcription
    prediction_rewards = []
    wer_rewards = []
    critical_word_rewards = []
    punctuation_rewards = []
    spelling_rewards = []
    
    for i, label in enumerate(labels):
        if i < len(aligned_predictions):
            pred = aligned_predictions[i]
            
            # Get text from label and prediction
            label_text = label.get("text", "").strip()
            pred_text = pred.get("text", "").strip()
            
            # Skip empty predictions or labels
            if not label_text or not pred_text:
                continue
            
            # Normalize case for comparison - SpeechBrain typically outputs uppercase
            label_text_norm = label_text.upper()
            pred_text_norm = pred_text.upper()
            
            # Calculate various reward components with case normalization
            text_reward = get_text_reward(label_text_norm, pred_text_norm)
            wer_reward = get_wer_reward(label_text_norm, pred_text_norm)
            critical_reward = check_critical_words(label_text_norm, pred_text_norm)
            punct_reward = check_punctuation(label_text, pred_text)  # Keep original case for punctuation
            spell_reward = check_spelling(label_text_norm, pred_text_norm)
            
            prediction_rewards.append(text_reward)
            wer_rewards.append(wer_reward)
            critical_word_rewards.append(critical_reward)
            punctuation_rewards.append(punct_reward)
            spelling_rewards.append(spell_reward)
    
    # Get average rewards for each component
    avg_prediction = sum(prediction_rewards) / max(1, len(prediction_rewards))
    avg_wer = sum(wer_rewards) / max(1, len(wer_rewards))
    avg_critical = sum(critical_word_rewards) / max(1, len(critical_word_rewards))
    avg_punctuation = sum(punctuation_rewards) / max(1, len(punctuation_rewards))
    avg_spelling = sum(spelling_rewards) / max(1, len(spelling_rewards))
    
    # Combine transcription-related rewards
    # Adjusted weights for SpeechBrain format - increase WER and edit distance importance
    w_edit = 0.35    # Edit distance-based similarity
    w_wer = 0.35     # Word Error Rate
    w_critical = 0.20  # Critical words accuracy
    w_punct = 0.05   # Punctuation accuracy
    w_spell = 0.05   # Spelling accuracy 
    
    # Combined transcription reward
    transcription_reward = (
        w_edit * avg_prediction +
        w_wer * avg_wer +
        w_critical * avg_critical +
        w_punct * avg_punctuation +
        w_spell * avg_spelling
    )
    
    # Compute gender reward using predicted_gender field
    gender_true = labels[0].get("gender", "").strip()
    gender_pred = getattr(response, "predicted_gender", None)
    gender_reward = get_gender_reward(gender_true, gender_pred)
    
    # Weights for final reward calculation
    alpha_transcription = 0.80  # Transcription is most important
    alpha_gender = 0.20        # Gender prediction
    
    # Combine all rewards
    total_reward = (
        alpha_transcription * transcription_reward +
        alpha_gender * gender_reward
    )
    
    # Log detailed reward components for debugging
    bt.logging.info(f"Reward components:")
    bt.logging.info(f"  Edit distance: {avg_prediction:.3f}")
    bt.logging.info(f"  WER: {avg_wer:.3f}")
    bt.logging.info(f"  Critical words: {avg_critical:.3f}")
    bt.logging.info(f"  Punctuation: {avg_punctuation:.3f}")
    bt.logging.info(f"  Spelling: {avg_spelling:.3f}")
    bt.logging.info(f"  Gender: {gender_reward:.3f}")
    bt.logging.info(f"  Total: {total_reward:.3f}")
    
    return total_reward