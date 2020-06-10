#MAKE GAME

# MAKE POLICY FOR AGENT  that uses the argmax of the random q funtion

# MAKE RANDOM- yes random - Q-learning function

import numpy as np
import time


class Game():
    def __init__(self):
        self.board = np.zeros(shape=(9,))
        self.done = False
    def step(self, team, action):
        self.board[action] = team
    def check(self, team):
        done = False
        if self.board[0] == team and self.board[1] == team and self.board[2] == team:
            done = True
        if self.board[3] == team and self.board[4] == team and self.board[5] == team:
            done = True
        if self.board[6] == team and self.board[7] == team and self.board[8] == team:
            done = True
        if self.board[0] == team and self.board[3] == team and self.board[6] == team:
            done = True
        if self.board[1] == team and self.board[4] == team and self.board[7] == team:
            done = True
        if self.board[2] == team and self.board[5] == team and self.board[8] == team:
            done = True
        if self.board[0] == team and self.board[4] == team and self.board[8] == team:
            done = True
        if self.board[2] == team and self.board[4] == team and self.board[6] == team:
            done = True
        if self.board[0] == team and self.board[1] == team and self.board[2] == team:
            done = True
        if 0 not in self.board:
            done = True
        self.done = done

class Agent():
    def __init__(self,team):
        self.team = team
    def Policy(self,state):
        return np.argmax(self.Q(state))

    def Q(self, state):
        x = np.ones(shape=(9,))
        y = np.empty(shape=(9,))
        print(x*y)
        return x*y

game = Game()
a =  Agent(1)
b = Agent(2)

while not game.done:
    aX = a.Policy(game.board)
    game.step(1,aX)
    forboard = []
    for i in range(9):
        x = game.board[i]
        if x == 0:
            forboard.append(' ')
        if x == 1:
            forboard.append('X')
        if x == 2:
            forboard.append('O')
    print(f"""
{forboard[0]}#{forboard[1]}#{forboard[2]}
-----
{forboard[3]}#{forboard[4]}#{forboard[5]}
-----
{forboard[6]}#{forboard[7]}#{forboard[8]}
        """)
    time.sleep(3)
    bX = b.Policy(game.board)
    game.step(2, bX)
    forboard = []
    for i in range(9):
        x = game.board[i]
        if x == 0:
            forboard.append(' ')
        if x == 1:
            forboard.append('X')
        if x == 2:
            forboard.append('O')
    print(f"""
{forboard[0]}#{forboard[1]}#{forboard[2]}
-----
{forboard[3]}#{forboard[4]}#{forboard[5]}
-----
{forboard[6]}#{forboard[7]}#{forboard[8]}
        """)
    time.sleep(3)
    game.check(1)
    game.check(2)
