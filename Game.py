import numpy as np


games = 2


class Game():
    def __init__(self):
        self.state = np.zeros(shape=(9,))
        self.over = False
    def step(self, action, team):
        self.state[action] = team

    def check(self, state, team):
        for i in range(2):
            team = i + 1
            done = False
            if self.state[0] == team and self.state[1] == team and self.state[2] == team:
                done = True
            if self.state[3] == team and self.state[4] == team and self.state[5] == team:
                done = True
            if self.state[6] == team and self.state[7] == team and self.state[8] == team:
                done = True
            if self.state[0] == team and self.state[3] == team and self.state[6] == team:
                done = True
            if self.state[1] == team and self.state[4] == team and self.state[7] == team:
                done = True
            if self.state[2] == team and self.state[5] == team and self.state[8] == team:
                done = True
            if self.state[0] == team and self.state[4] == team and self.state[8] == team:
                done = True
            if self.state[2] == team and self.state[4] == team and self.state[6] == team:
                done = True
            if self.state[0] == team and self.state[1] == team and self.state[2] == team:
                done = True
            if team == 1 and done == True:
                return True, 1
            if team == 2 and done == True:
                return True, 2
            else:
                return False, 0

class Agent():
    def __init__(self, team):
        self.team = team
    def Policy(self, state):
        actions = self.Q(state, self.team)
        action = np.argmax(actions)
        while state[action] != 0:
            actions[action] = None
            actions = self.Q(state, self.team)
            action = np.argmax(actions)
        return action

    def Q(self, state, team):
        x = np.random.uniform(0, 1, size=(9,))
        return x


a = Agent(1)
b = Agent(2)


for i in range(games):
    game = Game()
    team = 1
    winner = 0
    done = False
    while not game.over:
        if team == 1:
            action = a.Policy(game.state)
        if team == 2:
            action = b.Policy(game.state)
        game.step(action, team)
        forboard = []
        for i in range(9):
            if game.state[i] == 0:
                forboard.append(' ')
            if game.state[i] == 1:
                forboard.append('X')
            if game.state[i] == 2:
                forboard.append('O')


        print(f"{forboard[0]}|{forboard[1]}|{forboard[2]}")
        print("######")
        print(f"{forboard[3]}|{forboard[4]}|{forboard[5]}")
        print("######")
        print(f"{forboard[6]}|{forboard[7]}|{forboard[8]}")
        print("    ")
        done, teamThatWon = game.check(game.state,1)

        if done == True:
            winner = teamThatWon
            AA += 1
            print("GAME WON BY X")
            break
        if done == True and teamThatWon == 0:
            print("GAME ENDS IN TIE")
            break
        done, teamThatWon = game.check(game.state,2)

        if done == True:
            winner = teamThatWon
            BA += 1
            print("GAME WON BY O")
            break
        if done == True and teamThatWon == 0:
            print("GAME ENDS IN TIE")
            break

        if team == 1:
            team = 2
        elif team == 2:
            team = 1
