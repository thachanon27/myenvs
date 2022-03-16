import gym
import numpy as np
from gym import spaces
from gym.spaces import Discrete, Box
from scipy.stats import poisson
from random import randint, choice


class InventoryEnv(gym.Env):
    def __init__(self, config={}):
        self.l = config.get("lead time", 2)
        self.storage_capacity = 4000
        self.order_limit = 200
        self.step_count = 0
        self.max_steps = 40
        self.holding_cost = 5.0
        self.loss_goodwill = 10.0

        self.max_value = 100.0  # price and cost max ที่ 100
        self.max_mean = 200  # demand

        self.inv_dim = max(1, self.l)  # inv_dim = 2
        space_low = self.inv_dim * [0]
        space_high = self.inv_dim * [self.storage_capacity]
        space_low += 1 * [0]
        space_high += [
            200,
        ]
        self.observation_space = spaces.Box(
            low=np.array(space_low),
            high=np.array(space_high),
            dtype=np.float32
        )

        # Action is between 0 and 1, representing order quantity from
        # 0 up to the order limit.
        '''  self.action_space = spaces.Box(
            low=np.array([0]),
            high=np.array([1]),
            dtype=np.float32)
        '''
        self.action_space = Discrete(2)


        self.state = None
        self.reset()

    #   def _normalize_obs(self):
    #       obs = np.array(self.state)
    #       obs[:self.inv_dim] = obs[:self.inv_dim] / self.order_limit
    #       obs[self.inv_dim] = obs[self.inv_dim ] / 200
    #      return obs

    def reset(self):

        self.step_count = 0

        mean_demand = np.random.rand() * 200

        self.state = np.zeros(2 + 1)  ###แทน inv_dim = 2
        self.state[2] = mean_demand  ###แทน inv_dim = 2

        # return self._normalize_obs()
        return self.state

    def break_state(self):
        inv_state = self.state[: self.inv_dim]
        mu = self.state[self.inv_dim]  # mu = demand
        return inv_state, mu

    def step(self, action):
        h = 15  # holding cost
        k = 10  # Lost of good Will
        p = 100
        c = 80
        lt = 2
        invdim = 2
        stepcount = 0
        maxsteps = 40
        #beginning_inv_state, mu = self.break_state()
        beginning_inv_state = self.state[: 2]  #self.inv_dim = 2
        mu = self.state[2]  ##self.inv_dim = 2
        #action = np.clip(action[0], 0, 1)
        #action = int(action* 200)   #self.order_limit = 200
        #action = np.array([0,200])
        #action = 200
        done = False

        #available_capacity = self.storage_capacity - np.sum(beginning_inv_state)
        #assert available_capacity >= 0
        action = randint(0, 1)     ##อันนี้ให้ลองสุ่ม action เพื่อดูว่า รันออกมั้ย แต่พอไป link กับ a3c ให้เอา บรรทัดนี้ออก เพราะเป็น fn choose action จะสุ่มจาก prob policy ให้
        #print("action : ", action)
        buys = action*200  #self.order_limit = 200
        # If lead time is zero, immediately
        # increase the inventory
        if lt == 0:
            self.state[0] += buys
        on_hand = self.state[0]
        demand_realization = np.random.poisson(mu)

        # Compute Reward
        sales = min(on_hand,
                    demand_realization)  # ถ้า on_hand น้อยกว่า demand ก็ขายแค่ = on hand
        sales_revenue = p * sales
        overage = on_hand - sales
        underage = max(0, demand_realization
                       - on_hand)
        purchase_cost = c * buys
        holding = overage * h
        penalty_lost_sale = k * underage
        reward = sales_revenue \
                 - purchase_cost \
                 - holding \
                 - penalty_lost_sale

        # Day is over. Update the inventory
        # levels for the beginning of the next day
        # In-transit inventory levels shift to left
        self.state[0] = 0
        if invdim > 1:
            self.state[: invdim - 1] \
                = self.state[1: invdim]
        self.state[0] += overage  # Inventory ที่ หัก demand ออกไปแล้ว
        # Add the recently bought inventory
        # if the lead time is positive
        if lt > 0:
            #print(type(self.state[self.l - 4]))   #type = float64
            #buy2 = np.array(buys, dtype = float)
            #buys2 = np.clip(buys, 0, 1)
            self.state[lt - 4] = buys  # Buy แล้ว ของเข้า period ไหน

        self.step_count += 1
        if stepcount >= maxsteps:
            done = True

        # Normalize the reward
        reward = reward / 10000
        info = {
            "demand realization": demand_realization,
            "sales": sales,
            "underage": underage,
            "overage": overage,
        }
        print("Step :", self.step_count)
        print("On_hand :", on_hand)
        print("demand : ", demand_realization)
        print("period buy :", self.state[1:lt])
        print("Reward : ", reward)
        print("---------------------------------------------")
        #print("obs : ", self.state)
        # return self._normalize_obs(), reward, done, info
        return self.state, reward, done, info



def get_action_from_benchmark_policy(env):
    h = 5
    k = 10
    p = 100
    c = 80
    inv_state, mu = env.break_state()
    cost_of_overage = h
    cost_of_underage = p - c + k
    critical_ratio = np.clip(
        0, 1, cost_of_underage
              / (cost_of_underage + cost_of_overage)
    )
    horizon_target = int(poisson.ppf(critical_ratio,
                         (len(inv_state)+1) * mu))
    #deficit = max(0, env.order_limit)
    deficit = max(0, horizon_target - np.sum(inv_state))
    buy_action = min(deficit, env.order_limit)
    return [buy_action / env.order_limit]



if __name__ == "__main__":
    np.random.seed(100)
    env = InventoryEnv()
    episode_reward_avgs = []
    episode_total_rewards = []
    for i in range(100):
        print(f"Episode: {i+1}")
        initial_state = env.reset()
        done = False
        ep_rewards = []
        while not done:
            # action = env.action_space.sample()
            action = get_action_from_benchmark_policy(env)
            # print("Action: ", action)
            state, reward, done, info = env.step(action)
            # print("State: ", state)
            ep_rewards.append(reward)
        total_reward = np.sum(ep_rewards)
        reward_per_day = np.mean(ep_rewards)
        # print(f"Total reward: {total_reward}")
        # print(f"Reward per time step: {reward_per_day}")
        episode_reward_avgs.append(reward_per_day)
        episode_total_rewards.append(total_reward)
        print(
            f"Average daily reward over {len(episode_reward_avgs)} "
            f"test episodes: {np.mean(episode_reward_avgs)}. "
            f"Average total epsisode reward: {np.mean(episode_total_rewards)}")  
            
