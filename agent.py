import gymnasium as gym
import kymnasium as kym
import numpy as np
import pickle
from typing import Any, Dict

'''
kymnasium.Agent를 상속하여
자신만의 에이전트를 구현
'''
class YourAgent(kym.Agent):
    def __init__(self):
        self.policy = {}

    def act(self, observation: Any, info: Dict):
        loc = np.where(observation >= 1000)
        if len(loc[0]) > 0:
            r, c = loc[0][0], loc[1][0]
            d = int(observation[r, c] - 1000)
            return self.policy.get((r, c, d), 2)
        return 2

    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        agent = cls()
        with open(path, 'rb') as f:
            agent.policy = pickle.load(f)
        return agent

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

def train():
    '''
    Grid Crossing 환경은 다음과 같이 생성
    '''
    env = gym.make(
        id='kymnasium/GridWorld-Crossing-26x26',
        render_mode='rgb_array',
        bgm=False
    )
    '''
    여기서부터는 이 환경에 대해서 에이전트를 훈련시키는 코드를
    자유롭게 작성
    '''
    obs, _ = env.reset()
    V = np.zeros((26, 26, 4)) 
    phi = 1e-4 
    gamma = 0.99 

    def get_next_state_reward(r, c, d, action):
        if action == 0: 
            return r, c, (d - 1) % 4, -0.1
        elif action == 1: 
            return r, c, (d + 1) % 4, -0.1
        else: 
            dr = [0, 1, 0, -1]
            dc = [1, 0, -1, 0]
            nr, nc = r + dr[d], c + dc[d]
            
            if not (0 <= nr < 26 and 0 <= nc < 26) or obs[nr, nc] == 250:
                return r, c, d, -0.1
            if obs[nr, nc] == 900:
                return nr, nc, d, -100
            if obs[nr, nc] == 810:
                return nr, nc, d, 100
            return nr, nc, d, -1
    
    while True:
        delta = 0
        for r in range(26):
            for c in range(26):
                if obs[r, c] == 250 or obs[r, c] == 900 or obs[r, c] == 810:
                    continue
                for d in range(4):
                    v_old = V[r, c, d]
                    
                    q_list = []
                    for a in [0, 1, 2]:
                        nr, nc, nd, reward = get_next_state_reward(r, c, d, a)
                        q_list.append(reward + gamma * V[nr, nc, nd])
                    
                    V[r, c, d] = max(q_list) 
                    delta = max(delta, abs(v_old - V[r, c, d])) 
        
        if delta < phi: 
            break
            
    agent = YourAgent()
    for r in range(26):
        for c in range(26):
            for d in range(4):
                q_list = []
                for a in [0, 1, 2]:
                    nr, nc, nd, reward = get_next_state_reward(r, c, d, a)
                    q_list.append(reward + gamma * V[nr, nc, nd])
                agent.policy[(r, c, d)] = np.argmax(q_list)
    
    agent.save('trained_agent.pkl')

if __name__ == "__main__":
    train()
    
    trained_agent = YourAgent.load('trained_agent.pkl')
    
    kym.evaluate(
        env_id='kymnasium/GridWorld-Crossing-26x26',
        agent=trained_agent,
        bgm=True
    )