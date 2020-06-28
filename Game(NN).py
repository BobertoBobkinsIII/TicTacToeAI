import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
import time
import random
import pickle
games = 10000


class Game():
  def __init__(self, board=None):
    # killeen allow an existing board to instantiate this Game.
    if board is None:
      self.board = np.zeros(shape=(9,))
    else:
      self.board = board

  def __str__(self):
    board = [[' ', 'X', 'O'][idx] for i, idx in enumerate(self.board.astype(int))]
    str = ''
    for i in board:
      str = str + i
    return str

  def step(self, team, position):
    # self.board[position] = team
    # killeen: this is also a large part of the problem. You're not really saving the old state if
    # you just modify the existing game.
    board = self.board.copy()
    board[position] = team
    return Game(board)

  def checkWin(self, player):
    board = self.board.reshape(3, 3)
    p = board == player
    return (np.any(np.all(p, axis=0))          # |
            or np.any(np.all(p, axis=1))       # --
            or np.all(np.diag(p))              # \
            or np.all(np.diag(np.fliplr(p))))

  def showBoard(self):
    board = [[' ', 'X', 'O'][idx] for i, idx in enumerate(self.board.astype(int))]
    return """\
 {} | {} | {}
---+---+---
 {} | {} | {}
---+---+---
 {} | {} | {}

""".format(*board)

  def giveReward(self, player):
    """Given that *player* is the agent being rewarded, return the reward and whether the game is over.

    :param player:
    :returns: (done, reward) tuple.
    :rtype:

    """
    if player == 1:
      opp = 2
    elif player == 2:
      opp = 1
    else:
      raise ValueError

    if self.checkWin(player):  # Won
      return True, 1
    elif self.checkWin(opp):
      return True, 0
    elif np.any(self.board == 0):
      return False, 0         # still game to play
    # elif self.checkWin(player) == False and 0 not in self.board:  # Tie
    #   return True, 0
    else:  # In game
      return True, 0.5          # killeen: tie game




class Agent():
  def __init__(self, team):
    self.team = team
    self.qTable = {}
    self.alpha = 0.9 # 0.075
    self.gamma = .97
    self.model = Sequential()
    self.model.add(Dense(9, activation='relu'))
    self.model.add(Dense(1024, activation='relu'))
    self.model.add(Dense(1024, activation='relu'))
    self.model.add(Dense(9, activation='softmax'))
    self.model.compile(loss='mse', optimizer='Adam')

  def Policy(self, state):
      # killeen: I changed this back to zeros. In
                                                      # theory, it's fine either way.

    x = self.model.predict(state.board).copy()  # killeen:
    print(x)
    print(self.model.predict(state.board).shape)
    x = x.reshape((9,))
    # killeen: this shouldn't be necessary. Worst case, it's causing problems.
    # if x.shape != (9,):
    #   self.qTable[str(state)] = np.zeros(shape=(9,))
    # x = self.qTable[str(state)]

    # killeen: this is an alternative to the below
    x[state.board != 0] = -np.inf  # killeen: sets all unavailable action values to -inf
    action = np.random.choice(np.flatnonzero(x == x.max()))  # killeen: choose randomly from among

    # killeen: this might be fine, but it's pretty verbose. Writing concise code is not just a
    # matter of showing off your cleverness. It forces code to be more modular, readable, and
    # robust.

    # action = np.argmax(x)
    # if state.board[action] != 0:
    #   x[np.argmax(x)] = -np.inf
    #   action = np.argmax(x)
    # while state.board[action] != 0:
    #   x[np.argmax(x)] = -np.inf
    #   action = np.argmax(x)
    # sameQ = []
    # for i in range(len(x)):
    #   if x[action] == x[i]:
    #     if x[action] != -np.inf:
    #       sameQ.append(i)
    # if len(sameQ) >= 2:
    #   x[random.choice(sameQ)] += .1  # killeen: this is your problem. The policy shouldnt' be modifying the Q tables.

    return action

  def UpdateQ(self, oldState, action, Newstate, done, reward):
      pred = self.model.predict(oldState)
      index = np.argmax(pred)
      x = (reward+self.gamma*(np.max(pred)))
      pred[index] = x
      model.train_on_batch(oldState, pred)


def load(path):
  with open(path, 'rb') as file:
    qtable = pickle.load(file)
  return qtable


game = Game()
a = Agent(1)
b = Agent(2)
a.qTable[str(game)] = np.ones(shape=(9,))
b.qTable[str(game)] = np.ones(shape=(9,))
# a.qTable = load('AQtable.pkl')
# b.qTable = load('BQtable.pkl')
print('STARTING GAMES')
for i in range(games):
  game = Game()
  done = False
  Amove = 1
  Bmove = 1      # oldState  action  Newstate done reward
  transitions = [[None, None, None, None, None], [None, None, None, None, None]]
  wins = [0, 0, 0]              # tie games, p1, p2

  if i % 100 == 0:
    print('GAME:{}, Ties: {}, P1: {}, P2: {}'.format(i, *wins))          # killeen, printing number of wins

  while not done:               # killeen: This should ensure there's more moves to make.
    print(game.showBoard())
    if Amove != 1:
      done, reward = game.giveReward(1)
      # oppdone, _ = game.giveReward(2)  # killeen: done should be same for both right?
      # if oppdone:
      #   reward = 0
      #   done = True
      transitions[0][2] = game
      transitions[0][3] = done
      transitions[0][4] = reward
      a.UpdateQ(*transitions[0])  # killeen: more concise, less chance of index typo, easier to read.
      if done:
        if reward == 1:
          wins[1] += 1
        elif reward == 0.5:
          wins[0] += 1
        else:
          wins[2] += 1              # killeen: update the win stats
        break

    transitions[0][0] = game
    print(game.board)
    move = b.Policy(game)       # killeen: this now returns the move itself, not the qtable.
    transitions[0][1] = move
    game = game.step(2, move)   # killeen: this is updating the variable but not changing the old state now.
    # killeen: also, changed the team making the move. It's b's turn.
    if i % 100 == 0:
        print(f'GAME:{i}')
        print(game.showBoard())
        time.sleep(1)
    Amove += 1

    done, oppreward = game.giveReward(2)  # need to get new done/reward state for the new board, (reward for opponent)
    if done:
      # print("X WON")
      if oppreward == 1:
        wins[2] += 1
      elif oppreward == 0.5:
        wins[0] += 1
      else:
        wins[1] += 1              # killeen: update the win stats

      # update player 2
      transitions[1][2] = game
      transitions[1][3] = done
      transitions[1][4] = oppreward  # killeen: give new reward
      b.UpdateQ(*transitions[1])

      # update player 1
      done, reward = game.giveReward(1)  # need to get the game-ending reward for p1
      transitions[0][2] = game
      transitions[0][3] = done
      transitions[0][4] = reward
      a.UpdateQ(*transitions[0])
      break

    ################################################################################
    # killeen: copy everything but swap players
    ################################################################################
    print(game.showBoard())

    if Bmove != 1:
      done, reward = game.giveReward(2)
      transitions[1][2] = game
      transitions[1][3] = done
      transitions[1][4] = reward
      b.UpdateQ(*transitions[1])

    transitions[1][0] = game
    move = a.Policy(game)
    transitions[1][1] = move
    game = game.step(1, move)
    if i % 100 == 0:
        print(f'GAME:{i}')
        print(game.showBoard())
        time.sleep(1)
    move += 1

    done, oppreward = game.giveReward(1)
    if done:
      # print("X WON")
      if oppreward == 1:
        wins[1] += 1
      elif oppreward == 0.5:
        wins[0] += 1
      else:
        wins[2] += 1

      # update player 1
      transitions[0][2] = game
      transitions[0][3] = done
      transitions[0][4] = oppreward
      b.UpdateQ(*transitions[0])

      # update player 1
      done, reward = game.giveReward(1)
      transitions[1][2] = game
      transitions[1][3] = done
      transitions[1][4] = reward
      a.UpdateQ(*transitions[1])
      break

  print(game.showBoard())
with open('AQtable.pkl', 'wb') as file:
  pickle.dump(a.qTable, file)

with open('BQtable.pkl', 'wb') as file:
  pickle.dump(b.qTable, file)
