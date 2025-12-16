
from vali_objects.plagiarism.plagiarism_events import PlagiarismEvents
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import numpy as np
from vali_objects.vali_config import ValiConfig


class FollowPercentage(PlagiarismEvents):

  def __init__(self, plagiarist_id: str):
    super().__init__(plagiarist_id, "follow")
  

  def score(self, plagiarist_trade_pair: str, victim_key: tuple[str, str]):
    """
    Args:
        victim_key: tuple of (victim hotkey, victim trade pair)
    """

    plagiarist_orders = PlagiarismEvents.positions[(self.plagiarist_id, plagiarist_trade_pair)]
    victim_orders = PlagiarismEvents.positions[victim_key]
    event_key = (self.plagiarist_id, plagiarist_trade_pair, victim_key[0], victim_key[1])
    
    if event_key in self.time_differences:
      differences = self.time_differences[event_key]

    differences = FollowPercentage.compute_time_differences(plagiarist_orders, victim_orders)
    self.time_differences[event_key] = differences
    percent_of_follow = FollowPercentage.compute_follow_percentage(differences, victim_orders)
    
    plagiarism_key = (self.plagiarist_id, plagiarist_trade_pair, victim_key[0], victim_key[1])
    self.metadata[plagiarism_key] = {"victim": victim_key[0],
                                    "victim_trade_pair": victim_key[1],
                                    "type": self.name,
                                    "score": percent_of_follow
                                    }
  
  @staticmethod
  def compute_time_differences(plagiarist_orders: list, victim_orders: list):
    """
    Args:
        plagiarist_orders: list of states from state_dict which have the cumulative leverages of miner positions
    """

    time_resolution = ValiConfig.PLAGIARISM_MATCHING_TIME_RESOLUTION_MS
    time_window = ValiConfig.PLAGIARISM_ORDER_TIME_WINDOW_MS

    differences = []
    i = j = 0

    # Looks through all victim orders to collect all possible plagiarist orders that follow them
    while i < len(victim_orders) and j < len(plagiarist_orders):

        difference = plagiarist_orders[j]["start"] - (victim_orders[i]["start"])
        # Difference is within the follow time window and greater than 10 seconds
        if difference <= time_window and difference >= ValiConfig.PLAGIARISM_MINIMUM_FOLLOW_MS:
            differences.append(difference / time_resolution)

            i += 1
            j = 0
        else:
            
            j+= 1

    return differences


  @staticmethod
  def average_time_lag(plagiarist_orders=None, victim_orders=None, differences=None):
    """
    Args:
        plagiarist_orders: list of states from state_dict which have the cumulative leverages of miner positions
        differences: list of time differences between followed orders of plagiarist and victim in terms of the plagiarism time resolution
    """

    # Must provide orders or differences
    if differences is None and plagiarist_orders is not None and victim_orders is not None:
      differences = FollowPercentage.compute_time_differences(plagiarist_orders, victim_orders)
    elif differences is None:
      return 0

    avg_difference = int(sum(differences)/len(differences)) if len(differences) > 0 else 0

    return avg_difference

  @staticmethod
  def compute_follow_percentage(differences: list, victim_orders: list):
    """
    Args:
        victim_orders: list of states from state_dict which have the cumulative leverages of miner positions
        differences: list of time differences between followed orders of plagiarist and victim in terms of the plagiarism time resolution

    """

    if len(victim_orders) > 0:
      return len(differences)/len(victim_orders)
    else:
      return 0


class LagDetection(PlagiarismEvents):
  
  def __init__(self, plagiarist_id: str):
    super().__init__(plagiarist_id, "lag")

  
  def score(self, plagiarist_trade_pair: str, victim_key: tuple[str, str]):
    """
    Args:
        victim_key: tuple of (victim hotkey, victim trade pair)
    """
    lag_score = self.score_direct(plagiarist_trade_pair, victim_key)
    
    plagiarism_key = (self.plagiarist_id, plagiarist_trade_pair, victim_key[0], victim_key[1])

    self.metadata[plagiarism_key] = {"victim": victim_key[0],
                                    "victim_trade_pair": victim_key[1],
                                    "type": self.name,
                                    "score": lag_score
                                }
    
  def score_direct(self, plagiarist_trade_pair: str, victim_key: tuple[str, str]):
    """
    Args:
        victim_key: tuple of (victim hotkey, victim trade pair)
    """
    plagiarist_score = CopySimilarity.score_direct(self.plagiarist_id, plagiarist_trade_pair, victim_key[0], victim_key[1])
    victim_score = CopySimilarity.score_direct(victim_key[0], victim_key[1], self.plagiarist_id, plagiarist_trade_pair)

    lag_score = plagiarist_score / victim_score if victim_score > 0 else 0
    return lag_score


class CopySimilarity(PlagiarismEvents):
  
  def __init__(self, plagiarist_id: str):
    super().__init__(plagiarist_id, "single")


  def score(self, plagiarist_trade_pair: str, victim_key: tuple[str, str]):
    """
    Args:
        victim_key: tuple of (victim hotkey, victim trade pair)
    """

    similarity = CopySimilarity.score_direct(self.plagiarist_id, plagiarist_trade_pair, victim_key[0], victim_key[1])
    plagiarism_key = (self.plagiarist_id, plagiarist_trade_pair, victim_key[0], victim_key[1])

    self.metadata[plagiarism_key] = {"victim": victim_key[0],
                                    "victim_trade_pair": victim_key[1],
                                    "type": self.name,
                                    "score": similarity
                                    }
  
  @staticmethod
  def score_direct(plagiarist_id: str, plagiarist_trade_pair: str, victim_id: str, victim_trade_pair: str):

    plagiarist_vector = PlagiarismEvents.rasterized_positions[(plagiarist_id, plagiarist_trade_pair)]
    victim_vector = PlagiarismEvents.rasterized_positions[(victim_id, victim_trade_pair)]

    plagiarist_positions = PlagiarismEvents.positions[(plagiarist_id, plagiarist_trade_pair)]
    victim_positions = PlagiarismEvents.positions[(victim_id, victim_trade_pair)]

    differences = None
    event_key = (plagiarist_id, plagiarist_trade_pair, victim_id, victim_trade_pair)
    if event_key in PlagiarismEvents.time_differences:
      differences = PlagiarismEvents.time_differences[event_key]

    time_lag = FollowPercentage.average_time_lag(plagiarist_positions, victim_positions, differences=differences)
    if event_key in PlagiarismEvents.copy_similarities:
      similarity = PlagiarismEvents.copy_similarities[event_key]
    elif time_lag > 0:
      
      similarity = cosine_similarity([plagiarist_vector[time_lag:]], [victim_vector[:-time_lag]])[0][0]

    else:

      similarity = cosine_similarity([plagiarist_vector], [victim_vector])[0][0]

    PlagiarismEvents.copy_similarities[event_key] = similarity
    return similarity
  

class TwoCopySimilarity(PlagiarismEvents):
  
  def __init__(self, plagiarist_id: str):
    super().__init__(plagiarist_id, "two")


  def score_all(self, plagiarist_trade_pair: str):
    single_pair_similarities = {}
    for miner_id in PlagiarismEvents.miner_ids:

      event_key = (self.plagiarist_id, plagiarist_trade_pair, miner_id, plagiarist_trade_pair)
      if miner_id != self.plagiarist_id:
        victim_key = (miner_id, plagiarist_trade_pair)
        if event_key in self.copy_similarities:
          single_pair_similarities[victim_key] = self.copy_similarities[event_key]
        else: 
          single_pair_similarities[victim_key] = CopySimilarity.score_direct(self.plagiarist_id, plagiarist_trade_pair, miner_id, plagiarist_trade_pair)

    self.score(plagiarist_trade_pair, single_pair_similarities)

  
  def score(self, plagiarist_trade_pair: str, single_pair_similarities: dict):
    """
    Args:
        single_pair_similarities: dict of cosine similarities between 
          (plagiarist_id, plagiarist_trade_pair) and (victim_id, vicitim_trade_pair)
    """


    top_pairs = heapq.nlargest(2, single_pair_similarities.items(), key=lambda x: x[1])
    two_n_avg = np.average([x[1] for x in top_pairs])

    for pair in top_pairs:
      plagiarism_key = (self.plagiarist_id, plagiarist_trade_pair, pair[0][0], pair[0][1])
      self.metadata[plagiarism_key] = {"victim": pair[0][0],
                                      "victim_trade_pair": pair[0][1],
                                      "type": self.name,
                                      "score": two_n_avg
                                        }
    return self.metadata


class ThreeCopySimilarity(PlagiarismEvents):
  
  def __init__(self, plagiarist_id: str):
    super().__init__(plagiarist_id, "three")

  def score_all(self, plagiarist_trade_pair: str):

    single_pair_similarities = {}
    for miner_id in PlagiarismEvents.miner_ids:

      event_key = (self.plagiarist_id, plagiarist_trade_pair, miner_id, plagiarist_trade_pair)
      if miner_id != self.plagiarist_id:
        victim_key = (miner_id, plagiarist_trade_pair)
        if event_key in self.copy_similarities:
          single_pair_similarities[victim_key] = self.copy_similarities[event_key]
        else:
          single_pair_similarities[victim_key] = CopySimilarity.score_direct(self.plagiarist_id, plagiarist_trade_pair, miner_id, plagiarist_trade_pair)

    self.score(plagiarist_trade_pair, single_pair_similarities)


  def score(self, plagiarist_trade_pair: str, single_pair_similarities: dict):
    """
    Args:
        single_pair_similarities: dict of cosine similarities between 
          (plagiarist_id, plagiarist_trade_pair) and (victim_id, vicitim_trade_pair)
    """

    top_pairs = heapq.nlargest(3, single_pair_similarities.items(), key=lambda x: x[1])
    three_n_avg = np.average([x[1] for x in top_pairs])
    for pair in top_pairs:
      plagiarism_key = (self.plagiarist_id, plagiarist_trade_pair, pair[0][0], pair[0][1])
      self.metadata[plagiarism_key] = {"victim": pair[0][0],
                                      "victim_trade_pair": pair[0][1],
                                      "type": self.name,
                                      "score": three_n_avg
                                        }