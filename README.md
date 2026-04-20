# Minimax V1
I uploaded the first version using **minimax** and $\alpha-\beta$ **pruning**.

The code is workable by test but still needs refinement.

1. Some parameters regarding the `evaluate` function and `action_priority` need to be tuned.(Perhaps Machine learning can help)
2. After some test against itself, I found that with search step 2(I know it's very small, I just use it for a simple test) it will always split the stacks and maintaining a tie situation till exceeding the maximum turns(300 turns). I think some encouragement on applying merge can be used.
  <img width="927" height="327" alt="image" src="https://github.com/user-attachments/assets/dc068fa6-22c1-4799-9f86-1cdb0a5a0e24" />
3. I planned to set up an easily-performing agent, but I haven't really come up with one. This can test the ability for our agent to fight against others.
