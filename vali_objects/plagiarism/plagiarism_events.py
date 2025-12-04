from abc import abstractmethod

from vali_objects.utils.reporting_utils import ReportingUtils

class PlagiarismEvents:
  positions = {}
  rasterized_positions = {}
  time_differences = {}
  copy_similarities = {}
  miner_ids = []
  trade_pairs = []
    
  def __init__(self, plagiarist_id: str, name: str):
    self.metadata = {} # Dictionary of data for plagiarism events
    self.plagiarist_id = plagiarist_id
    self.name = name


  def score_all(self, plagiarist_trade_pair: str):

    for miner_id in PlagiarismEvents.miner_ids:

      if miner_id != self.plagiarist_id:
        
        victim_key = (miner_id, plagiarist_trade_pair)
        self.score(plagiarist_trade_pair, victim_key)
    return

  @abstractmethod
  def score(self):
    """Each plagiarism_definitions subclass must override and define their own score method"""
    pass

  def summary(self) -> dict:
    return self.metadata

  @staticmethod
  def set_positions(positions: dict, miner_ids: list[str], trade_pairs: list[str], current_time=None, lookback_window=None, time_resolution=None):
    """
    Args:
        positions: cumulative leverage positions for each miner as a dictionary with (miner hotkey, miner trade pair) as the key
    """

    PlagiarismEvents.positions = positions
    PlagiarismEvents.miner_ids = miner_ids
    PlagiarismEvents.trade_pairs = trade_pairs
    PlagiarismEvents.rasterized_positions = ReportingUtils.rasterize(positions, current_time=current_time, lookback_window=lookback_window, time_resolution=time_resolution)

  
  
  @staticmethod
  def clear_plagiarism_events():
    PlagiarismEvents.positions = {}
    PlagiarismEvents.rasterized_positions = {}
    PlagiarismEvents.time_differences = {}
    PlagiarismEvents.copy_similarities = {}
    PlagiarismEvents.miner_ids = []
    PlagiarismEvents.trade_pairs = []

