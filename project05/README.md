# Introduction
Creating functions that perform the smith-waterman method for local alignment. The method is composed of initializing
an array, scoring the matrix and then performing the traceback. The results are two sequences with optimal local alignment.

# Successes
We understood the way the smith-waterman method worked. Conceptually the method for local alignment was relatively easy to follow. We also were able to create thoughout pseudocode which provided a decent blueprint.

# Struggles
Though we were able to create the cal_score function relatively quick, the traceback function was difficult to write. We also had conflicting schedules which prevented us from meeting up synchronously as often as we would have wanted.

# Personal Reflections
## Group Leader
One issue I ran into was that I forgot the 'padding' at the start of the matrix, which allows for us to have a valid start position - so I ended up with a natural 'off-by-1' error. I ended up getting a different result than what the expected output was, but it was an equally good alignment. If I had more time, I was going to figure out how to return all equally good alignments as well (not sure if this is typical for Smith-Waterman but would be a good exercise), and I was trying to figure out how the expected output was reached based on what we prioritize. I assume that if there were an identical score between up/left/diag, we'd prioritize diag, but whether up/left should take precedence, I'm not sure if it really matters. This was taken care of in the `max` implementation since it should select the first one if there's a tie.

## Other member (Allen Benavidez)
I felt that conceptually this might be the simplest out of the previous project we had. I felt that the scoring function was fairly easy however once I encoutered the traceback function, I repeatedly ran into issues possibly from an infinite loop that I could not figure out. Though I ran into a variety of issues in my code, I am feeling significantly better about my coding ability.

# Generative AI Appendix
As per the syllabus
