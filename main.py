# ####################################################
# DE2-COM2 Computing 2
# Individual project
#
# Title: MAIN
# Authors: Oliver Colebourne
# ####################################################

"""
    Solution to tetriling problem.

    The Tetris function was imported by performance testing code, which randomly 
    generated a target matrix, ran the solver then analysed the solution.

    In the assessment of the coursework the solver was tested with three different 
    targets of varying size and density.

"""

# example target, a 20by20 0.6 density:
target = [
    [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0],
    [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1],
    [1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
    [1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1]
]


from copy import deepcopy, copy

def Tetris(target):

    solve = ProjSolver(target)
    M = solve.bestSolution
    
    return M

class ProjSolver:
    """
        Main class containing all methods to solve tetriling problem, for target T
        - Initialise with target matrix as argument
        - self.bestSolution should be called to provide the solution

        Each method contains description of its use and overview of how it works.
        Lines are commented for more in-depth explanation of how each algorithm works.
    """
    def __init__(self, T):
        print('Solver intialised')
        self.T = T
        self.tRows = len(T)     #number of rows
        self.tCols = len(T[0])  #number of columns
        self.S = []             #create empty solution matrix
        for i in range(self.tRows): 
            self.S.append([(0,0)]*self.tCols)
        self.TCopy = deepcopy(self.T)
        self.n, self.m = 0,0    #used to track position - n for column, m for row
        self.pieceCount = 0     #for pieceID
        self.forcing = 0
        self.piecePath = [0]*4
        self.bestSolution = 0
        self.bestAccuracy = 0

        # TREE OF PIECES
        # coordinates relative from top left corner of starting position using following reference axis.
        # lowest level of the tree contains the shapeIDs.
        #  ---> +ve n
        # |
        # |
        # V
        # +ve m
        # pieces ordered prioritising left and top weighted pieces.
        self.piecesTree = Node('start', [   
            Node((0,1),[
                Node((-1,1),[
                    Node((-1,2),[Node('19')]),
                    Node((-2,1),[Node('5')]),
                    Node((0,2),[Node('14')]),
                ]),
                Node((1,0),[
                    Node((-1,1),[Node('16')]),
                    Node((0,2),[Node('10')])
                ]), 
                Node((0,2),[
                    Node((-1,2),[Node('8')]),
                    Node((1,2),[Node('4')]),
                    Node((1,1),[Node('12')]),
                ]),
                Node((1,1),[
                    Node((1,2),[Node('17')]),
                    Node((2,1),[Node('11')]),
                    Node((-1,1),[Node('13')])
                ]),
                
            ]),
            Node((1,0),[
                Node((1,1),[
                    Node((2,1),[Node('18')]),
                    Node((1,2),[Node('6')]),   
                    ]),
                Node((2,0),[
                    Node((1,1),[Node('15')]),
                    Node((0,1),[Node('7')]),
                    Node((2,1),[Node('9')]),
                    ]),

            ])
            ])

        # Used to check the neighbours of a position. 
        # Defining here allowed me to experiment with including diagonals aswell.
        self.pointEdges = [ 
            (0,1),
            (1,0),
            (0,-1),
            (-1,0),
            ] 

        self.runSolver()

    def runSolver(self):
        """
            Main control for solver. Ran from __init__
            1. WalkMatrix and fit pieces for first pass.
            2. Pass again with forcing enabled then save the solution.
            3. Finish if best accuracy is 100%.
            4. If smaller matrix run flipMatrix method (doubles time so only for smaller ones).
        """
        self.walkMatrix()
        self.forcing = 1
        self.n, self.m = 0,0 #reset pointer to top left
        self.walkMatrix()
        self.saveSolution()
        if self.bestAccuracy == 1.0:
            return
        if (self.tCols + self.tRows) <= 500: #if less or equal to 250 by 250 run flipMatrix 
            self.flipMatrix()

        
    def walkMatrix(self):
        """
            Main walk/scanning method
            - Walks along the matrix from the top left corner until it finds a 1.
            - Runs the optimum piece fitting method depending on size of the matrix.
            - After piece is fitted, return back to this method and continue searching from the same place, for the next 1.

            Iterative rather than recursive to avoid recursion limits (without changing them).
        """
        pointer = self.TCopy[self.m][self.n]
        while True:
            # FIND NEXT 1   
            while not pointer: #loop while value in TCopy at current position is 0
                self.n += 1
                if self.n == self.tCols: #if at end of row, move to first postion of next row
                    self.n=0
                    self.m+=1
                if self.m==self.tRows: #if at end of matrix finish
                    return
                pointer = self.TCopy[self.m][self.n] #update pointer

            # PLACE PIECE 
            if self.forcing: #if forcing mode run the simple fitPiece function
                self.fitPiece(self.piecesTree.children, 1)

            # Designed so it could be varied easily in testing
            elif (self.tCols + self.tRows) <= 2000: #if size less than or equal to 1000 by 1000 run fitPiece2.
                self.piecePath[0]=(self.n,self.m) #set first point in piecePath
                self.bestPiece = [0,0,0] #reset bestPiece
                self.fitPiece2(self.piecesTree.children, 1, 1)

            else: #if larger run simple, faster, fitPiece method
                self.fitPiece(self.piecesTree.children, 1)
            pointer=0

    def fitPiece(self, currNode, level):
        """
            Depth-first search of piece tree. 
            - Recursive and greedy algorithm. Places first piece that fits then moves on.
            - Uses list of children at each node, comparing start point + child's relative coordinates to matrix.
            - If the point == 1 then then run the method again adding that child to the currNode.
            - At the lowest level of the tree where the .data is a string of the shapeID.
            - foundPiece then returned up the levels of the tree changing the relative values in the TCopy and S, solution, matrices
            - If no pieces are found then returns to walkMatrix.

            Fastest, lightest method with ~90-95% accuracy. Will solve a 1000x1000 in under 3seconds using this method.
            Orignally designed to be used as base method for large matrices except fitPiece2 accuracy gains are worth the extra time.
            Method is used when forcing for speed.
        """
        if type(currNode[0].data) == str: #picks up .data being a string ie that method has reached shapeID at bottom of tree and piece found
            foundPiece = currNode[0].data #foundPiece set to shapeID 
            self.pieceCount+=1
            self.TCopy[self.m][self.n] = 0 #remove start point from TCopy
            self.S[self.m][self.n] = (int(foundPiece), self.pieceCount) #place piece at start point in solution
            return foundPiece #pass shapeID up a level of tree

        for i, j in enumerate(currNode): #enumerate through current Node's children
            (N,M) = j.data
            if (self.m+M) < 0 or (self.n+N) <0 or (self.m+M) >= self.tRows or (self.n+N) >= self.tCols: 
                #check that start point + child's relative coords are within matrix, before indexing (else it raises index error)
                continue #skip this point if outside matrix
            
            #FORCING & LOWEST LEVEL
            if self.forcing and level == 3: 
                #look for empty space in solution matrix (placing the excess piece)
                point = self.S[(self.m)+M][(self.n)+N] 
            
            #NORMAL
            else:
                point = self.TCopy[(self.m)+M][(self.n)+N] #look in TCopy at start point (from walk) + relative coords of that child
            
            if point == 1 or point == (0,0): #if its 1 (or (0,0) when forcing on third level) then continue into that Node
                foundPiece = self.fitPiece(currNode[i].children, level+1) #recur fitPiece with the current Node and level updated.
                if type(foundPiece) == str: #if foundPiece is returned as a string (meaning it found a piece)
                    self.TCopy[self.m+M][self.n+N] = 0 #set point to 0 in TCopy
                    self.S[self.m+M][self.n+N] = (int(foundPiece), self.pieceCount) #update solution matrix for current point
                    return foundPiece #go up to next level passing the foundPiece shapeID string up

        #if no acceptable points were found at a current node then go up a level or return to the walkMatrix
        return   

    def fitPiece2(self, currNode, level, startingLoop=0):
        """
            Fit function with scoring to choose best piece
            - Same recursive tree as fitPiece is used.
            - At each level piecePath is updated with coodinates of the current point, before progressing to next level.
            - As pieces are found their pieceScore is calculated with the scorePiece method.
            - pieceScore is compared to the bestPiece's score and if better, bestPiece is updated with the new shapeID, score and path.
            - The method then continues 'preorder' tree traversal to find other pieces.
            - Once all pieces that fit have been scored and compared, the bestPiece is then placed in the S, solution and removed from TCopy.
            - If no pieces are found then returns to walkMatrix.

        """
        if type(currNode[0].data) == str: #picks up .data being a string ie that method has reached shapeID at bottom of tree and piece found
            foundPiece = currNode[0].data #foundPiece set to pieceID
            self.setPoints(self.piecePath,2) #set proposed piece's points to 2 in TCopy so they don't interfere with scoring
            pieceScore = self.scorePiece(self.piecePath) #score found for proposed piece
            self.setPoints(self.piecePath,1) #set proposed piece's points back to 1 in TCopy
            if pieceScore > self.bestPiece[1]: 
                self.bestPiece = [foundPiece, pieceScore, copy(self.piecePath)] #if new best piece score replace bestPiece
            return 
        
        for i, j in enumerate(currNode): #enumerate through current node's children
            (N,M) = j.data
            if (self.m+M) < 0 or (self.n+N) <0 or (self.m+M) >= self.tRows or (self.n+N) >= self.tCols: #if outside matrix skip point
                continue
            point = self.TCopy[self.m+M][self.n+N] #find value of TCopy at child's point
            if point == 1:
                self.piecePath[level]=(self.n+N,self.m+M) #set piecePath coordinate for the current level
                self.fitPiece2(currNode[i].children, level+1) #recur method for that node's children, increasing level by 1

        if not startingLoop: #if not at the highest level go up
            return
        elif self.bestPiece == [0,0,0]: #if no pieces were found return to walkMatrix
            return

        self.pieceCount += 1
        for i in self.bestPiece[2]: #loop through best piece's points
            N =  i[0]
            M =  i[1]
            self.TCopy[M][N] = 0 #remove points from TCopy
            self.S[M][N] = (int(self.bestPiece[0]), self.pieceCount) #place piece in solution matrix
        return   

    def setPoints(self, points, value): 
        """
            Takes list of pairs of coordinates and sets each coord in the TCopy matrix to a provided value
        """
        for i in points: #loop through tuples of point coords
            N =  i[0]
            M =  i[1]
            self.TCopy[M][N] = value

    def scorePiece(self, points):
        """
            Score piece, counting how many zeros are around it and checking for isolated 1s
            - For each point in the inputted piecePath, checks horizontal and vertical neighbouring points.
            - Adds each point it looks at to checkedList to save time and avoid double counting.
            - If a neighbouring point is a 0 +1 to the score.
            - If a neighbouring point is a 1 then check the points around that neighbour to ensure its not isolated (surrounded by 3 zeros).
            - Return the piece's score.

            Finding pieces with the most surrounding zeros will ensure that the piece placed fits well with the edge of the target.
            This means its less likely to leave awkward areas where pieces can't fit.
            Checking for isolated 1s, ie 1s left on their own after a piece is placed, means the correct piece is more likely to be selected.
        """
        score = 0
        checkedList = []
        for i in points: #cycle through the list of tuples, coords of the piece's path
            for j in self.pointEdges: #cycle through the relative coords for the neighbours around each point
                edgeN =  i[0] + j[0]
                edgeM =  i[1] + j[1]
                if (edgeN, edgeM) in checkedList: #skip if already checked
                    continue
                checkedList.append((edgeN,edgeM)) 
                if edgeM<0 or edgeN<0 or edgeM>=self.tRows or edgeN>=self.tCols: 
                    #check if point is in matrix, before indexing to avoid index error
                    score+=1 #if point is off the matrix treat this as a zero then skip
                    continue
                point = self.TCopy[edgeM][edgeN] #look at point in TCopy
                if point == 0: #if neighbouring point in TCopy is a zero then +1 to score and continue
                    score+=1
                elif point == 1: #if neighbouring point in TCopy is a 1 check around it to ensure not isolated
                    surroundZeros=0 
                    for k in self.pointEdges: #cycle through neighbouring points of the 1
                        surroundN =  edgeN + k[0]
                        surroundM =  edgeM + k[1] 
                        if surroundM<0 or surroundN<0 or surroundM>=self.tRows or surroundN>=self.tCols:
                            #check if point is in matrix, before indexing to avoid index error
                            surroundZeros+=1 #if point is off the matrix treat this as a zero
                        elif self.TCopy[surroundM][surroundN]==0: #if point is a zero +1 to surroundZeros
                            surroundZeros+=1 
                    if surroundZeros >2: #after checking around 1 if there are more than 2 zeros then 1 is isolated 
                        if self.bestPiece == [0,0,0]: #if this is the first (/potentially only solution) then ignore the isolated 1
                            continue
                        else: #set score to 0 and return - look for a better piece
                            score = 0 
                            return score
        return score

    def flipMatrix(self):
        """
            Flip the matrix and run solver again
            - Target is flipped horizontally and solver is run.
            - Solution is then flipped back, and the shapeIDs are corrected with their flipped equivalents.
            - The new solution is then ran through saveSolution method, choosing the new best solution

            Due to the nature of the problem sometimes the best piece in terms of scoring is not the best solution. 
            For large problems this averages out but for small matrixes this creates a large inconsistancy in the accuracy score.
            Therefore a way to run the same problem multiple times producing different solutions with the same method was needed:
            Randomising which piece was picked for fitPiece2 was likely to reduce accuracy.
            Reordering the tree on the fly for fitPiece, without having multiple trees, is very hard.
            Method needed to use existing solving methods without repeating them.
        """
        self.TCopy = [x[::-1] for x in self.T] #copy T as self.TCopy but flipped horizontally. Each row flipped with list comprehension.
        self.n, self.m = 0,0 #reset variables for a new walk/solve
        self.pieceCount = 0 
        self.forcing = 0
        self.S = []
        for i in range(self.tRows): #create empty solution matrix
            self.S.append([(0,0)]*self.tCols)
        self.walkMatrix()

        self.forcing = 1 #walk again forcing pieces in where there's three
        self.n, self.m = 0,0 
        self.walkMatrix()

        self.S = [x[::-1] for x in self.S] #flip solution matrix horizontally, back to original orientation 
        self.correctShapeIDs() #correct shapeIDs to flipped equivalents
        self.saveSolution() #score solution and save if better than bestSolution 
        return
    
    def correctShapeIDs(self):
        """
            Correct the shapeIDs after flipping the matrix horizontally
            - Replaces each shapeID with the horizontally flipped equivalent
        """
        self.flippedIDs = { #dictionary of flipped shapeIDs - allows for quick referencing
            4:8,
            5:11,
            6:10,
            7:9,
            8:4,
            9:7,
            10:6,
            11:5,
            12:14,
            13:13,
            14:12,
            15:15,
            16:18,
            17:19,
            18:16,
            19:17
        }
        m,n = 0,0
        while True: #run through the whole solution matrix
                point = self.S[m][n]
                if point[0] != 0: #if current 'point' is part of a piece
                    self.S[m][n] = (self.flippedIDs[point[0]],point[1]) #change the solution tuple at that point to one with flipped shapeID
                n+=1 #move right
                if n == self.tCols: #at end of row go to start of next row
                    n=0
                    m+=1
                if m==self.tRows: #at bottom of matrix return
                    return 

    def saveSolution(self):
        """
            Score and save the current self.S solution
            - Find the accuracy of the current self.S solution. 
            - Compare it to the current best solution's accuracy and replace bestSolution if it's an improvement.
        """
        missing, excess = self.calcMissExcess() #find number of missing and excess blocks using utils
        accuracy = 1 - (missing+excess)/(self.tRows*self.tCols) #not actual accuracy but saves time of counting 1s in target
        if accuracy > self.bestAccuracy:
            self.bestAccuracy = accuracy # update bestAccuracy
            self.bestSolution = deepcopy(self.S) # update bestSolution making sure to give it new memory location
        return

    def calcMissExcess(self):
        """
            Calculate the missing and excess blocks in the solution
        """
        missing = 0
        excess = 0

        for i in range(self.tRows):
            for j in range(self.tCols):
                if self.T[i][j] == 0 and self.S[i][j] != (0,0): #if 0 in target check if a piece was placed in solution
                    excess += 1 #+1 to excess if it was
                elif self.T[i][j] == 1 and self.S[i][j] == (0,0): #if 1 in target check if no piece placed in solution
                    missing += 1 #+1 to missing if no piece placed
        return missing, excess


class Node(object):
    """
        Class used for pieces tree.
        - For each node a list of children is created. 
        - Each child is also an instance of the Node class. 
        - Each node instance contains data which is which is referenced by the .data variable.
    """
    def __init__(self, data='start', children=None):
        self.data = data
        self.children = []
        if children is not None: #run add_child method for each child
            for child in children:
                self.children.append(child)
