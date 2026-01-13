---
title: Minesweeper
description:
tags:
  - "#projects"
  - tech
date: 2023-06-28
---
# A New Obsession: Building a Minesweeper AI

This project was part of my journey through Harvard's CS50 AI course, and it turned out to be one of the most amazing projects I've ever worked on.

To be honest, I didn't even know the exact rules of **Minesweeper** before I started. I spent about a week just playing the game to understand its logic. I quickly started enjoying it so much that it's become a favorite pastime, right up there with my previous obsession: Sudoku.
## Demonstration and Code

Here's a quick video of the AI in action:

![Minesweeper](https://youtu.be/GfYEJIYUaNk?si=wxryAZkupYo4xg2O)


And here is the complete source code on GitHub:
* **[Minesweeper Solver Source Code](https://github.com/suryaanshrai/minesweeper)**
## What I Learned

This project was a deep dive into **knowledge-based systems**. It taught me how an AI can manage a "knowledge base" and use logical inference to derive *new* knowledge from what it already knows. It gave me a completely new perspective on how artificial intelligence can be used to master rule-based games.

> [!info]
> Check out the official [CS50 AI Minesweeper Project Page](https://cs50.harvard.edu/ai/projects/1/minesweeper/) for more details on the assignment and the course.



# The Gist of the Project

I'd say the first three steps were very obvious from the project's [description page itself](https://cs50.harvard.edu/ai/projects/1/minesweeper/), the key to this project was figuring out the [[Minesweeper#4. Handling the newly updated knowledge base|4th step]].


### 1. Representing Knowledge

Instead of complex propositional logic, the project uses a simple and powerful `Sentence` class. A "sentence" in this AI is just two things:
1.  A **set of cells** (e.g., `{(0, 1), (0, 2), (1, 1)}`)
2.  A **count** of how many mines are in that set.

So, if I clicked a cell and it revealed a "2", I would add a new sentence to my AI's **Knowledge Base (KB)** representing its 8 neighbors and the count of 2.

### 2. The AI's Logic Loop

The `MinesweeperAI` class follows a clear, safety-first strategy:

1.  **Find a Safe Move:** First, I implemented the `make_safe_move` function. This function simply picks up a safe move from all the possible ones at the moment and pops it from the set (`self.safes`). (*Had to change the self.safes from list to set in the starter code so that the make_safe_move operation maybe more time and space efficient*)
2.  **Make a Random Move:** If there are no guaranteed safe moves left, the AI has to guess. My `make_random_move` function picks a move at random, making sure not to pick a square that's already been played or one that is a *known mine*.

### 3. The "Magic": Inferring New Knowledge

The real intelligence of the project is in the `add_knowledge` function. This function is the AI's "brain" and gets called every time a new move is made.

Here’s what it does:

1.  **Adds New Info:** It takes the `(cell, count)` from the latest move and adds it as a new `Sentence` to the knowledge base.
2.  **Finds Obvious Answers:** It immediately checks the KB for simple conclusions. If any sentence has a `count` of 0, it marks all cells in that set as **safe**. If a sentence's `count` is equal to the number of cells in its set (e.g., 3 mines in a 3-cell set), it marks them all as **mines**.
3.  **Infers by Subtraction:** This is the coolest part. The AI loops through its entire knowledge base, comparing every sentence to every other sentence. If it finds two sentences where one is a subset of the other, it can infer a *new sentence*.

    * **Example:**
        * The AI knows: `Sentence A = {(0,1), (0,2)} = 1`
        * And it knows: `Sentence B = {(0,1), (0,2), (0,3)} = 2`
    * My code can then infer a new, simpler sentence by "subtracting" A from B:
        * **New Inferred Sentence C:** `{(0,3)} = 1`

    The AI now knows for a *fact* that cell `(0,3)` is a mine, without ever having to guess. This new sentence is added back into the knowledge base, and the whole inference process repeats until no new conclusions can be drawn. This loop allows the AI to chain logical deductions together to solve the board.

### 4. Handling the newly updated knowledge base
If any new knowledge is found, i.e., the knowledge base is updated, then the inference system is activated again to infer new knowledge from the updated knowledge base. This is a recursive step and is done until the knowledge base cannot infer any new knowledge from itself. 

This way, if any new safe moves are possible they are added to the knowledge base at once.
# End notes
This project was, in a word, a joy. It was one of those rare assignments that stops feeling like an "assignment" and just becomes an intriguing puzzle you *need* to solve. I found myself very occupied by it, thinking about inference logic and knowledge bases even when I was away from the computer.

The enthusiasm for seeing the AI make a smart, logical move—or for figuring out the bug in my inference loop—was the real driver. I honestly forgot it was part of a course or that there were scores involved. It was just a truly fun endeavor, a perfect blend of a classic game and a fascinating AI challenge. 
