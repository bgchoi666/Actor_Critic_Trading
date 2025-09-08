# <model description>

# actor_critic_reinfo
Advatage Actor-Critic trains a model that outputs optimal reinfo values. The state is the data from Algorithms 2-1, 4-1, and 5-1, and the reward environment is the return based on the reinfo actions of the Bong and Bong-sa of the randomly selected ensemble models within the given algorithm range. The weights are updated using (expected return, critic model value) and (actor model action probability).

# actor_critic_reinfo2
Advance Actor-Critic trains a model that outputs optimal reinfo values.
The state is the data from Algorithms 2-1, 4-1, and 5-1. The reward environment is the return based on the reinfo actions of the ensemble models randomly selected within the given algorithm range. The sum over a given period is the return.
The difference from version 1 is that model information is added as input.

# actor_critic_reinfo3
Advatage Actor-Critic trains a model that outputs optimal reinfo values. The state represents the return trends for the previous 100 bars from Algorithms 2-1, 4-1, and 5-1. The new state represents the return that changes according to the reinfo action, from which the reward and expected return are calculated.

# actor_critic_reinfo4
Advatage Actor-Critic trains a model that outputs optimal reinfo values. The state represents the return trends for the previous 100 bars from Algorithms 2-1, 4-1, and 5-1. The new state represents the return that changes according to the reinfo action, from which the reward and expected return are calculated.
