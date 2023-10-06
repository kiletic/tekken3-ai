class MatchStatus:
  ACTIVE = 0
  FINISHED = 1

class Observation:
  def __init__(self):
    self.frame = None
    self.match_status = MatchStatus.ACTIVE
    self.game_paused = 0
    self.player1_hp = 0
    self.player2_hp = 0
