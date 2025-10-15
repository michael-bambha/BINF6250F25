
  # Introduction
Creating functions that perform the smith-waterman method for local alignment. The method is composed of initializing
an array, scoring the matrix and then performing the traceback. The results are two sequences with optimal local alignment.

# Pseudocode
```
### Calc Diag
- Take in positions of 2 nucleotides, return match/mismatch score

### Cal_Score

- find positions of all of our adjacent up/diag/left
- calc scores of those based on current val
- find max score and the direction of the max score and return


create the matrix (N+1 x N+1)
- add zeros to both top row and left most column 

function cal_score(matrix, seq1, seq2, i, j, match, mismatch, gap):
   

    if letters of first two sequences match then diagonal value +1, if not then -1 and add to list
    Top value -2 and add to list
    Left value -2 and add to list.
    obtain max value in list
    if max value < 0 then set the max value to 0. 
    ^^ need some way to find which of the values returned the max (which direction)


Move one to the right in the row. If end of row then move to 2 column 1 + row number
if no more rows then return the matrix with score

### Traceback

- Find the maximum value in our matrix (np.unravel_index, np.argmax, np.max)
- From the max:
    - Find adjacent neighbors, left, up, diag
    - Return position of the maximum of those neighbors
    - If all are 0, we are done


    aligned_seq1 = ""
    aligned_seq2 = ""

     aligned_seq1 += "seq1[current_row]"
        aligned_seq2 += seq2[current_col]
        current_row, current_col -= 1
        
    aligned_seq1 += "-"
        aligned_seq2 += seq2[current_col]
```

# Successes
We understood the way the smith-waterman method worked. Conceptually the method for local alignment was relatively easy to follow. We also were able to create thoughout pseudocode which provided a decent blueprint.

# Struggles
Though we were able to create the cal_score function relatively quick, the traceback function was difficult to write. We also had conflicting schedules which prevented us from meeting up synchronously as often as we would have wanted.

# Personal Reflections
## Group Leader
Group leader's reflection on the project

## Other member (Allen Benavidez)
I felt that conceptually this might be the simplest out of the previous project we had. I felt that the scoring function was fairly easy however once I encoutered the traceback function, I repeatedly ran into issues possibly from an infinite loop that I could not figure out. Though I ran into a variety of issues in my code, I am feeling significantly better about my coding ability.

# Generative AI Appendix
As per the syllabus
