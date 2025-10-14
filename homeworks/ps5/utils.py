from copy import copy
class Error(Exception):
    pass

# General game state object to work with multiple games
class game_state:
    def __init__(self,board,parent=None):
        self.board = board
        self.parent = parent
        self.v = None
        self.optimal_play = None
        
        # If the parent is not None, the player is inferred from the player of the parent
        if self.parent != None:
            self.player = (parent.player)%2 + 1
        else:
            self.player = 1        
    
    def __str__(self):
        return self.board.__str__() + '\n'
    
    def successors(self):
        succ_states = []
        
        for succ_board in self.board.successors(self.player):
            succ_state = game_state(succ_board,self)
            succ_states += [succ_state]
        
        return succ_states
    
    def terminal_check(self):
        return self.board.terminal_check()
    
    def score(self):
        return self.board.score()

    
# Tic-tac-toe board as an example of a simple and solvable game
class tic_tac_toe_board:
    def __init__(self, moves):
        if len(moves) != 9:
            raise Error
        self.moves = moves
    
    def __str__(self):
        return '{0} | {1} | {2}\n--+---+--\n{3} | {4} | {5}\n--+---+--\n{6} | {7} | {8}'.format( \
       self.moves[0],self.moves[1],self.moves[2],self.moves[3],self.moves[4],self.moves[5],self.moves[6],self.moves[7],self.moves[8])
    
    def terminal_check(self):
        # Empty Board
        if not ' ' in self.moves:
            return True
        
        for symbol in ['o','x']:
            # Rows
            for row in range(0,3):
                if self.moves[3*row] == symbol and self.moves[3*row] == self.moves[3*row+1] and self.moves[3*row] == self.moves[3*row+2]:
                    return True
            
            # Columns
            for col in range(0,3):
                if self.moves[col] == symbol and self.moves[col] == self.moves[col+3] and self.moves[col] == self.moves[col+6]:
                    return True
            
            # Diagionals
            if self.moves[0] == symbol and self.moves[0] == self.moves[4] and self.moves[0] == self.moves[8]:
                return True
            if self.moves[2] == symbol and self.moves[2] == self.moves[4] and self.moves[2] == self.moves[6]:
                return True
            
        return False
    
    def score(self):
        if not self.terminal_check():
            return 0
        
        # Rows
        for row in range(0,3):
            if self.moves[3*row] == 'x' and self.moves[3*row] == self.moves[3*row+1] and self.moves[3*row] == self.moves[3*row+2]:
                return 1
            if self.moves[3*row] == 'o' and self.moves[3*row] == self.moves[3*row+1] and self.moves[3*row] == self.moves[3*row+2]:
                return -1
        
         # Columns
        for col in range(0,3):
            if self.moves[col] == 'x' and self.moves[col] == self.moves[col+3] and self.moves[col] == self.moves[col+6]:
                return 1
            if self.moves[col] == 'o' and self.moves[col] == self.moves[col+3] and self.moves[col] == self.moves[col+6]:
                return -1
        
        # Diagionals
        if self.moves[0] == 'x' and self.moves[0] == self.moves[4] and self.moves[0] == self.moves[8]:
            return 1
        if self.moves[2] == 'x' and self.moves[2] == self.moves[4] and self.moves[2] == self.moves[6]:
            return 1
        if self.moves[0] == 'o' and self.moves[0] == self.moves[4] and self.moves[0] == self.moves[8]:
            return -1
        if self.moves[2] == 'o' and self.moves[2] == self.moves[4] and self.moves[2] == self.moves[6]:
            return -1
        
        # If we get here, there is a tie
        return 0
    
    def successors(self,player):
        # Crosses correspond to player 1, naughts to 2
        if self.terminal_check():
            raise Error
        
        succ_list = []
        
        for i in range(0,9):
            if self.moves[i] == ' ':
                succ_moves = copy(self.moves)
                
                if player == 1:
                    succ_moves[i] = 'x'
                else:
                    succ_moves[i] = 'o'
                
                succ_list += [tic_tac_toe_board(succ_moves)]
        
        return succ_list
    
