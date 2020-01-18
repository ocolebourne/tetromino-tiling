# Tetromino Tiling
Solver for tiling tetrominos into a randomly generated target grid
## The challenge
 
 ![Tetris pieces](/Images/shapes.jpg)

The challenge was to create an algorithm to find a tiling solution to a randomly generated target. The above tetris shapes were the tetriminos allowed to be used. Every target has a perfect solution (it is formed by placing pieces randomly without overlapping).

For example:
![Example task](/Images/example_task.jpg)

The target is provided in the form of a list of lists (ie each sub-list is a row of the grid). Where there is a 1 a piece should be placed. The solution should be provided where each point on the grid is a tuple - (0,0) for empty space or (pieceID, shapeID) where a piece was placed (where pieceID is the count of pieces placed and shapeID is the number assigned to each specific tetris piece, labelled above).

For example:
![Example task](/Images/example_task2.jpg)

## My solution

My solution used a recursive depth-first search (preorder traversal) through a pieces tree, each time a 1 was found. It dynamically changed between greedy for large grids - picking the first piece that fit from the tree - or a scoring system to place pieces surrounded by the most 0s and skipping pieces that left isolated 1s. 
