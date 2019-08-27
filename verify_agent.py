from spinup.utils.test_policy import load_policy
import matplotlib.pyplot as plt
import numpy as np

nb_steps = 250
init_angles = []
init_pos = []
init_rewards = []

env, get_action = load_policy('logging/test')


def get_reward_from_init_angle_of_trained_agent(init_angle):
    state, reward, finished = env.reset(init_angle), 0, False
    proposed_action = get_action(state)
    n = 1
    while not (finished):
        proposed_action = get_action(state)
        state, reward, finished, _ = env.step(proposed_action)
        n += 1
    return env.reward, n


for i in range(nb_steps):
    env.reset()
    init_pos.append(env.state)
    init_rewards.append(env.reward)
    init_angles.append([env.mssb_angle, env.mbb_angle])

init_pos.append([0., min(np.array(init_pos)[:, 1])])
init_pos = np.array(init_pos)

new_rewards = []
time = []
for i in range(nb_steps):
    reward, n = get_reward_from_init_angle_of_trained_agent(init_angles[i])
    new_rewards.append(reward)
    time.append(n)

new_rewards.append(0.)


plot_name = 'Old'
name = plot_name
plt.scatter(init_pos[:-1, 0], init_pos[:-1, 1], c=init_rewards, alpha=0.1)
plt.title(name)
# plt.savefig(name + '.pdf')
plt.show()

plt.scatter(init_pos[:, 0], init_pos[:, 1], c=new_rewards, alpha=0.1)
plot_name = 'New'
name = plot_name
plt.title(name)
# plt.savefig(name + '.pdf')
plt.show()