import numpy as np
import time


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
 {} | {} | {}
---+---+---
 {} | {} | {}
---+---+---
 {} | {} | {} """.format(*board)


    def giveReward(self, player):
        if self.checkWin(player):#Won
            return True, 1
        elif self.checkWin(player) == False and 0 not in self.board:#Tie
            return False, 0.5
        else:#In game
            return False, -1


class Agent():
    def __init__(self, team):
        self.team = team
        self.qTable = {}
        self.alpha = 0.1
        self.gamma = .95
    def policy(self, state):
        if state not in self.qTable:
            self.qTable[str(state)] = np.zeros(shape=(9,))
        x = self.qTable[str(state)]
        if state.board[np.argmax(x)] != 0:
            x[np.argmax(x)] = (np.min(x) - 1)

    def update(self, board_state, action, new_board_state, new_done, new_reward):
        if str(board_state) not in self.qTable:
            self.qTable[str(board_state)] = np.zeros(shape=(9,))
        if new_done:
            self.qTable[str(board_state)][action] = new_reward
        else:
            self.qTable[str(board_state)][action] = ((1 - self.alpha)* self.qTable[str(board_state)][action]+self.alpha * self.gamma * np.max(self.policy(new_board_state)))
a = Agent(1)
b = Agent(2)
for i in range(10):

    game = Game()
    done = False
    a.qTable[str(game)] = np.zeros(shape=(9,))
    b.qTable[str(game)] = np.zeros(shape=(9,))
    print(a.qTable)
    while not done:
        #if i+1 % 1 == 0:
        print(game.showBoard())
        time.sleep(2)
        oldState = game
        aMove = a.policy(game)
        game.step(1, aMove)
        done, reward = game.giveReward(1)
        print(str(oldState))
        a.update(str(oldState), aMove, str(game), done, reward)
        if game != done:
            if i+1 % 1 == 0:
                print(game.showBoard())
                time.sleep(2)
            oldState = game.board
            bmove = b.policy(game)
            game.step(2, bmove)
            done, reward = game.giveReward(2)
            a.update(oldState, bmove, game, done, reward)
