import numpy as np
import time
import random

games = 2



class Game():
    def __init__(self):
        self.board = np.zeros(shape=(9,))
    def __str__(self):
        board = [[' ', 'X', 'O'][idx] for i, idx in enumerate(self.board.astype(int))]
        str =  ''
        for i in board:
            str = str+i
        return str
    def step(self, team, position):
        self.board[position] = team


    def checkWin(self, player):
        board = self.board.reshape(3,3)
        p = board == player
        return (np.any(np.all(p, axis=0))          # |
              or np.any(np.all(p, axis=1))       # --
              or np.all(np.diag(p))              # \
              or np.all(np.diag(np.fliplr(p))))
    def showBoard(self):
        board = [[' ', 'X', 'O'][idx] for i, idx in enumerate(self.board.astype(int))]
        return """\
GAME:{}
 {} | {} | {}
---+---+---
 {} | {} | {}
---+---+---
 {} | {} | {}

 """.format(game, *board)


    def giveReward(self, player):
        if self.checkWin(player):#Won
            return True, 1
        elif self.checkWin(player) == False and 0 not in self.board:#Tie
            return True, 0.5
        else:#In game
            return False, -1


class Agent():
    def __init__(self, team):
        self.team = team
        self.qTable = {}
        self.alpha = 0.1
        self.gamma = .95
    def Policy(self, state):
        if str(state) not in self.qTable:
            self.qTable[str(state)] = np.zeros(shape=(9,))
        x = self.qTable[str(state)]
        action = np.argmax(x)
        if state.board[action] != 0:
            x[int(np.argmax(x))] = int((np.min(x)-1))
            action = np.argmax(x)
        while state.board[action] != 0:
            x[int(np.argmax(x))] = int((np.min(x)-1))
            action = np.argmax(x)
        return x
    def UpdateQ(self, oldState, action, Newstate, done, reward):
        print(str(oldState))
        print(str(Newstate))


        if str(Newstate) not in self.qTable:
            self.qTable[str(Newstate)] =  np.zeros(shape=(9,))
        if done:
            self.qTable[str(oldState)][action] = reward
        else:
            self.qTable[str(Newstate)] = ((1-self.alpha) * self.qTable[str(oldState)][action] + self.alpha * self.gamma * np.max(self.Policy(Newstate)))


game = Game()
a = Agent(1)
b = Agent(2)
a.qTable[str(game)] = np.zeros(shape=(9,))
b.qTable[str(game)] = np.zeros(shape=(9,))

for i in range(games):
    game = Game()
    done = False
    while not done:
        print(game.showBoard())
        time.sleep(1)
        oldstate = str(game)
        print(oldstate)
        move = np.argmax(a.Policy(game))
        game.step(1, move)
        done, reward = game.giveReward(1)
        a.UpdateQ(oldstate, move, game, done, reward)
        if not done:
            print(game.showBoard())
            time.sleep(1)
            oldstate = game
            move = np.argmax(b.Policy(game))
            game.step(2, move)
            done, reward = game.giveReward(2)
            b.UpdateQ(oldstate, move, game, done, reward)
    print(game.showBoard())
    print(b.qTable)
