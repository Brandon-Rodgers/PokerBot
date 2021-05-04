from functools import reduce

class ActionChecker:

  @classmethod
  def correct_action(self, players, player_pos, sb_amount, action, amount=None):
    if self.is_allin(players[player_pos], action, amount):
      amount = players[player_pos].stack + players[player_pos].paid_sum()
    elif self.__is_illegal(players, player_pos, sb_amount, action, amount):
      action, amount = "fold", 0
    return action, amount


  @classmethod
  def is_allin(self, player, action, bet_amount):
    if action == 'call':
      return bet_amount >= player.stack + player.paid_sum()
    elif action == 'raise':
      return bet_amount == player.stack + player.paid_sum()
    else:
      return False


  @classmethod
  def need_amount_for_action(self, player, amount):
    return amount - player.paid_sum()


  @classmethod
  def agree_amount(self, players):
    last_raise = self.__fetch_last_raise(players)
    return last_raise["amount"] if last_raise else 0


  @classmethod
  def legal_actions(self, players, player_pos, sb_amount, pot=0):
    min_raise = self.__min_raise_amount(players, sb_amount)
    max_raise = players[player_pos].stack + players[player_pos].paid_sum()
    BB = sb_amount*2
    if max_raise < min_raise:
      min_raise = max_raise
    return [
        { "action" : "fold" , "amount" : 0 },
        { "action" : "call" , "amount" : self.agree_amount(players) },
        { "action" : "raise", "amount" : { "min": min_raise, "2x": min_raise+BB, "3x": min_raise+BB*2, "4x": min_raise+BB*3, "5x": min_raise+BB*4, "6x": min_raise+BB*5,
                                          "7x": min_raise+BB*6, "8x": min_raise+BB*7, "9x": min_raise+BB*8, "10x": min_raise+BB*9, "11x": min_raise+BB*10, "12x": min_raise+BB*11,
                                          "13x": min_raise+BB*12, "14x": min_raise+BB*13, "15x": min_raise+BB*14, "16x": min_raise+BB*15, "17x": min_raise+BB*16, "18x": min_raise+BB*17, 
                                          "19x": min_raise+BB*18, "20x": min_raise+BB*19, "25x": min_raise+BB*24, "30x": min_raise+BB*29, "35x": min_raise+BB*34, "40x": min_raise+BB*39,
                                          "45x": min_raise+BB*44, "50x": min_raise+BB*49, "60x": min_raise+BB*59, "70x": min_raise+BB*69, "80x": min_raise+BB*79, "90x": min_raise+BB*89,
                                          "100x": min_raise+BB*99, "110x": min_raise+BB*109, "120x": min_raise+BB*119, "130x": min_raise+BB*129, "140x": min_raise+BB*139, "150x": min_raise+BB*149,
                                          "175x": min_raise+BB*174, "200x": min_raise+BB*199, "225x": min_raise+BB*224, "250x": min_raise+BB*249, "275x": min_raise+BB*274, "300x": min_raise+BB*299,
                                          "350x": min_raise+BB*349, "400x": min_raise+BB*399, "450x": min_raise+BB*449, "500x": min_raise+BB*499, "600x": min_raise+BB*599, "750x": min_raise+BB*749, 
                                          "900x": min_raise+BB*899, "1000x": min_raise+BB*999, "max": max_raise } }
    ]

  @classmethod
  def _is_legal(self, players, player_pos, sb_amount, action, amount=None):
    return not self.__is_illegal(players, player_pos, sb_amount, action, amount)

  @classmethod
  def __is_illegal(self, players, player_pos, sb_amount, action, amount=None):
    if action == 'fold':
      return False
    elif action == 'call':
      return self.__is_short_of_money(players[player_pos], amount)\
          or self.__is_illegal_call(players, amount)
    elif action == 'raise':
      return self.__is_short_of_money(players[player_pos], amount) \
          or self.__is_illegal_raise(players, amount, sb_amount)

  @classmethod
  def __is_illegal_call(self, players, amount):
    return amount != self.agree_amount(players)

  @classmethod
  def __is_illegal_raise(self, players, amount, sb_amount):
    return self.__min_raise_amount(players, sb_amount) > amount

  @classmethod
  def __min_raise_amount(self, players, sb_amount):
    raise_ = self.__fetch_last_raise(players)
    return raise_["amount"] + raise_["add_amount"] if raise_ else sb_amount*2

  @classmethod
  def __is_short_of_money(self, player, amount):
    return player.stack < amount - player.paid_sum()

  @classmethod
  def __fetch_last_raise(self, players):
    all_histories = [p.action_histories for p in players]
    all_histories = reduce(lambda acc, e: acc + e, all_histories)  # flatten
    raise_histories = [h for h in all_histories if h["action"] in ["RAISE", "SMALLBLIND", "BIGBLIND"]]
    if len(raise_histories) == 0:
      return None
    else:
      return max(raise_histories, key=lambda h: h["amount"])  # maxby

