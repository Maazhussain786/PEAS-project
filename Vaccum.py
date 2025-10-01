import random
import matplotlib.pyplot as plt

ACTIONS = ["SUCK","LEFT","RIGHT","UP","DOWN","NOP"]
class VacuumWorld:
    def __init__(self, width=3, height=3, dirt_spawn_prob=0.05, observability="full", max_steps=10):
        self.w = width
        self.h = height
        self.dirt_spawn_prob = dirt_spawn_prob
        self.observability = observability
        self.max_steps = max_steps
        self.reset()
    def reset(self):
        self.grid = [[random.choice([0,1]) for _ in range(self.w)] for __ in range(self.h)]
        self.agent_pos = (0,0)
        self.t = 0
        return self.observe()
    def in_bounds(self, x,y):
        return 0 <= x < self.w and 0 <= y < self.h

    def step(self, action):
        reward = -0.25  
        if action == "SUCK":
            x,y = self.agent_pos
            if self.grid[y][x] == 1:
                reward += 10
                self.grid[y][x] = 0
        elif action in ("LEFT","RIGHT","UP","DOWN"):
            dx = {"LEFT":-1,"RIGHT":1,"UP":0,"DOWN":0}[action]
            dy = {"LEFT":0,"RIGHT":0,"UP":-1,"DOWN":1}[action]
            nx,ny = self.agent_pos[0]+dx, self.agent_pos[1]+dy
            reward -= 1
            if self.in_bounds(nx,ny):
                self.agent_pos = (nx,ny)
            else:
                reward -= 0.5

        for y in range(self.h):
            for x in range(self.w):
                if self.grid[y][x]==0 and random.random()<self.dirt_spawn_prob:
                    self.grid[y][x] = 1
        self.t += 1
        done = (self.t>=self.max_steps)
        return self.observe(), reward, done

    def observe(self):
        x,y = self.agent_pos
        if self.observability=="full":
            return {"pos":self.agent_pos, "grid":[row[:] for row in self.grid]}
        else:
            return {"pos":self.agent_pos, "dirty_here":self.grid[y][x]==1} 
class RandomAgent:
    def act(self, obs):
        return random.choice(ACTIONS)

class ReflexAgentLocal:
    def act(self, obs):
        if obs.get("dirty_here",False):
            return "SUCK"
        return random.choice(ACTIONS)
    
class ReflexAgentFull:
    def __init__(self,w,h): 
        self.w=w; self.h=h
    def act(self, obs):
        x,y = obs["pos"]
        if obs["grid"][y][x]==1:
            return "SUCK"
        dirt_cells = [(xx,yy) for yy in range(self.h) for xx in range(self.w) if obs["grid"][yy][xx]==1]
        if not dirt_cells: return "NOP"
        dirt = min(dirt_cells, key=lambda p: abs(p[0]-x)+abs(p[1]-y))
        dx,dy = dirt[0]-x, dirt[1]-y
        if abs(dx)>0: return "RIGHT" if dx>0 else "LEFT"
        if abs(dy)>0: return "DOWN" if dy>0 else "UP"
        return "NOP"

def run_episode_with_logs(agent, env, name):
    obs = env.reset()
    total=0
    print(f"\n=== Running {name} Agent ===")
    steps=0
    while True:
        action = agent.act(obs)
        obs, reward, done = env.step(action)
        total += reward
        steps += 1
        print(f"Step {steps}: Action={action}, Reward={reward:.2f}, Pos={obs['pos']}, Total={total:.2f}")
        if done: break
    print(f"Final Total Reward for {name}: {total}\n")
    return total
def run_experiment():
    dirt_probs = [0.0, 0.05, 0.1, 0.2, 0.3]
    agents = {
        "Random": RandomAgent(),
        "Local Reflex": ReflexAgentLocal(),
        "Full Reflex": ReflexAgentFull(3,3)
    }
    results = {name:[] for name in agents}
    episodes = 20 

    for p in dirt_probs:
        for name,agent in agents.items():
            scores=[]
            for _ in range(episodes):
                env = VacuumWorld(width=3,height=3,dirt_spawn_prob=p,observability="full",max_steps=30)
                scores.append(run_episode(agent,env))
            results[name].append(sum(scores)/len(scores))

    for name,vals in results.items():
        plt.plot(dirt_probs, vals, marker="o", label=name)
    plt.xlabel("Dirt Spawn Probability")
    plt.ylabel("Mean Reward")
    plt.title("Vacuum World: Mean Reward vs Dirt Probability")
    plt.legend()
    plt.show()

def run_episode(agent, env):
    obs = env.reset()
    total=0
    while True:
        action = agent.act(obs)
        obs, reward, done = env.step(action)
        total += reward
        if done: break
    return total
if __name__=="__main__":
    env = VacuumWorld(width=3,height=3,dirt_spawn_prob=0.1,observability="full",max_steps=10)
    run_episode_with_logs(RandomAgent(), env, "Random")
    run_episode_with_logs(ReflexAgentLocal(), env, "Local Reflex")
    run_episode_with_logs(ReflexAgentFull(3,3), env, "Full Reflex")
    run_experiment()