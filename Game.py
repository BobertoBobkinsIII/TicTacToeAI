import numpy as np
import time
import random

games = 20



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
        print("CHECKING WIN")
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
        if x.shape != (9,):
            self.qTable[str(state)] = np.zeros(shape=(9,))
        x = self.qTable[str(state)]
        action = np.argmax(x)
        if state.board[action] != 0:
            x[np.argmax(x)] =  -8
            action = np.argmax(x)
        while state.board[action] != 0:
            x[np.argmax(x)] = -8
            action = np.argmax(x)
        return x
    def UpdateQ(self, oldState, action, Newstate, done, reward):
        if str(Newstate) not in self.qTable:
            self.qTable[str(Newstate)] = np.zeros(shape=(9,))
        if done:
            self.qTable[str(oldState)][action] = reward
        else:
            self.qTable[str(Newstate)][action] = ((1-self.alpha) * self.qTable[str(oldState)][action] + self.alpha * self.gamma * np.max(self.Policy(Newstate)))

game = Game()
a = Agent(1)
b = Agent(2)
a.qTable[str(game)] = np.zeros(shape=(9,))
b.qTable[str(game)] = np.zeros(shape=(9,))

for i in range(games):
    game = Game()
    done = False
    Amove = 1
    Bmove = 1      # old  action  Newstate done reward
    transitions = [[None,None,None,None,None],[None,None,None,None,None]]
    print(game.showBoard())
    while not done:
        if Amove != 1:
            print("UPDATING Q")
            done, reward = game.giveReward(1)
            transitions[0][2] = game
            transitions[0][3] = done
            transitions[0][4] = reward
            a.UpdateQ(transitions[0][0],transitions[0][1], transitions[0][2], transitions[0][3], transitions[0][4])
        transitions[0][0] = game
        move = np.argmax(a.Policy(game))
        transitions[0][1] = move
        game.step(1, move)
        print(game.showBoard())
        time.sleep(1)
        Amove += 1
        if done:
            print("GAME IS DONE")
            transitions[1][2] = game
            a.UpdateQ(transitions[0][0],transitions[0][1], transitions[0][2], transitions[0][3], transitions[0][4])
            b.UpdateQ(transitions[1][0],transitions[1][1], transitions[1][2], transitions[1][3], transitions[1][4])
        if not done:
            if Bmove != 1:
                print("UPDATING Q")
                done, reward = game.giveReward(2)
                transitions[1][2] = game
                transitions[1][3] = done
                transitions[1][4] = reward
                b.UpdateQ(transitions[1][0],transitions[1][1], transitions[1][2], transitions[1][3], transitions[1][4])
            transitions[1][0] = game
            move = np.argmax(b.Policy(game))
            transitions[1][1] = move
            game.step(2, move)
            print(game.showBoard())
            time.sleep(1)
            Bmove += 1
            if done:
                print("GAME IS DONE")
                transitions[0][2] = game
                a.UpdateQ(transitions[0][0],transitions[0][1], transitions[0][2], transitions[0][3], transitions[0][4])
                b.UpdateQ(transitions[1][0],transitions[1][1], transitions[1][2], transitions[1][3], transitions[1][4])

    print(b.qTable)
