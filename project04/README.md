# Project 04: Functional file parsing
Michael Bambha and Brooks Groharing

# Introduction
In this project, we implemented functions to assemble De Bruijn graphs from input data and re-assemble these into valid eulerian paths.

## Pseudocode

To build the graph:

- Slide in a window to map k-1-mer prefixes to suffixes. Create an adjacency list from each prefix node.
- For adding an edge we simply append to the list. Removing a node, we will remove the instance from the list.
- May also need to delete the key if no more edges remain


For traversing:
```{Python}
# Outer function to initiate recursive search
def print_eulerian_walk():
	for left_node in graph edges:
		save a copy of graph
		walk = eulerian_walk(left_node) # removes edges from graph
		restore graph to copy

		if walk != False, return it
	
	return False # default condition; no eulerian walks found.


# Inner recursive function
eulerian_walk(current_node, toured_nodes):
	add current_node to toured_nodes

	if node has outgoing edges:
		pick a random connected next_node
		remove edge from current_node to next_node from graph
		return eulerian_walk(current_node = next_node, toured_nodes)
	else:
		if graph has no remaining edges:
			return tour
		else:
			return False # we've hit a dead end
```

### Another possibility
Note that instead of true recursion, we could instead trace paths using a stack:
```
- Initialize stack to the valid starting node
- While the stack is non empty:
- If the chosen node has untraversed edges:
- Traverse forwards, picking a random edge
- Append chosen edge to the stack
- Else:
- Pop the node from the stack and append to tour list

```
The only thing that we might have to consider is that some graphs may not have a Eulerian circuit.
We could implement finding if the path/circuit exist, which we were told we could ignore for simplicity,
but if we ignore it and a circuit does not exist, we will get stuck by not choosing the correct
starting node.


## Successes

We successfully implemented the algorithm, and we also added a couple of extra functionality to our
class instance to check if a Eulerian path/circuit exists.

## Challenges

Recursion is always difficult to conceptualize, so implementing a recursive approach took some debugging.
Also, another issue was finding if a Eulerian path/circuit existed - the main challenge here was calculating
the in-degrees. I realized the in-degree calculation could potentially be off in the event that a node
has no outgoing edges but only incoming ones (anyone reviewing can feel free to point out a better way
to calculate these if you'd like). I'm not entirely sure it matters that much for this specific case since we only need to know
the starting node in cases where there is no circuit (and the starting node identification seems to work), but
it could still be improved.

## Reflection

### Group Leader - Michael

Building the graph was super easy, but traversing it takes a lot more work. Since I was hitting a dead end,
I decided to implement finding whether a Eulerian circuit / path exists. I know this wasn't a necessary part
of the assignment since we were told we could ignore it, but it was difficult to know whether I was getting
dead ends due to poor implementation or whether there is only one valid starting position to recover a Eulerian path.
I dove a bit into the `functools` library for the cached property decorator, which was cool, although I'm unsure
if this is the ideal way we would do this. (Cached property was used solely because checking if the graph has a Eulerian path would need to re-check if a circuit existed, which could be cached if called prior).Admittedly my OOP skills could use some work, and I was unsure what
the correct design pattern is for our case here. Since we need to remove edges, calling our `remove_edge` function
will modify our graph in-place. I know I could use `deepcopy` to get a full copy of our instance, but wasn't sure
if that would be ideal to do or not. 
Lastly, I understood the algorithm much better without doing it recursively, but the functions provided were 
intended for a recursive approach. I rarely use recursion so it's probably more so that I'm not used to it, but it
was good practice.

### Contributor - Brooks

As Michael notes, implementing the De Bruijn graph constructor was very straightforward--I think we designed and wrote it basically on the spot. Implementing the Eulerian walk functions using the methods suggested in the provided file proved more time consuming. 

For one thing, recursion is inherently a bit tricky to conceptualize and debug. Truthfully, our initial pseudocode wasn't quite fit for the task, and it took some trial and error to figure out the correct return cases.

I also struggled a bit with how best to store and interact with the graph during the recursive walk, since the assignment directed us to remove edges as they were traversed. After some deliberation I decided to just save a single copy of the graph before attempting a walk, modifying the De Bruijn object's copy while traversing it, and then restoring the stored graph using the unedited copy after. I don't love this as a solution, but it is at least simpler (and required less modifications to the provided function arguments) than, for example, passing a modified copy of the graph into every recursive function call.

## Generative AI Appendix

We did not use generative AI in this project.
