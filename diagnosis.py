import random
import matplotlib.pyplot as plt
DISEASES = ["flu","cold","allergy"]
PRIOR = {"flu":0.3, "cold":0.4, "allergy":0.3}

TESTS = {
    "fever": {"flu":0.9,"cold":0.6,"allergy":0.1},
    "cough": {"flu":0.7,"cold":0.8,"allergy":0.3},
    "sneeze":{"flu":0.2,"cold":0.3,"allergy":0.9},
}
TEST_COST = {"fever":-5,"cough":-5,"sneeze":-5}

class DiagnosisWorld:
    def __init__(self):
        self.true_disease = random.choices(DISEASES, weights=[PRIOR[d] for d in DISEASES])[0]
        self.tests_done = {}
        self.done = False

    def order_test(self, test):
        if test in self.tests_done:
            return 0, False
        prob = TESTS[test][self.true_disease]
        result = (random.random() < prob)
        self.tests_done[test] = result
        return TEST_COST[test], False

    def diagnose(self, disease):
        self.done = True
        if disease == self.true_disease:
            return 100, True
        else:
            return -100, True
def posterior(tests_done):
    probs = {}
    for d in DISEASES:
        p = PRIOR[d]
        for t,res in tests_done.items():
            like = TESTS[t][d] if res else (1-TESTS[t][d])
            p *= like
        probs[d] = p
    s = sum(probs.values())
    for d in probs: probs[d] /= s if s>0 else 1
    return probs

class RandomDiagnosisAgent:
    def act(self, world):
        if len(world.tests_done) < 1:
            return ("order_test", random.choice(list(TESTS.keys())))
        else:
            return ("diagnose", random.choice(DISEASES))

class GreedyAgent:
    def act(self, world):
        post = posterior(world.tests_done)
        best = max(post, key=post.get)
        if len(world.tests_done) < 1:
            return ("order_test", "fever")
        else:
            return ("diagnose", best)

class TestThenDiagnoseAgent:
    def __init__(self, max_tests=2): self.max_tests=max_tests
    def act(self, world):
        if len(world.tests_done) < self.max_tests:
            remaining = [t for t in TESTS if t not in world.tests_done]
            if remaining: return ("order_test", remaining[0])
        post = posterior(world.tests_done)
        best = max(post, key=post.get)
        return ("diagnose", best)

def run_episode_with_logs(agent, name):
    world = DiagnosisWorld()
    total=0
    steps=0
    print(f"\n=== Running {name} Agent ===")
    while not world.done:
        action = agent.act(world)
        if action[0]=="order_test":
            reward, done = world.order_test(action[1])
            total += reward
            steps+=1
            print(f"Step {steps}: Ordered test={action[1]}, Result={world.tests_done[action[1]]}, Reward={reward}, Posterior={posterior(world.tests_done)} Total={total}")
        else:
            reward, done = world.diagnose(action[1])
            total += reward
            steps+=1
            print(f"Step {steps}: Diagnose={action[1]}, True={world.true_disease}, Reward={reward}, Total={total}")
    print(f"Final Total Reward for {name}: {total}\n")
    return total
def run_experiment(episodes=50):
    agents = {
        "Random": RandomDiagnosisAgent(),
        "Greedy": GreedyAgent(),
        "TestThenDiagnose": TestThenDiagnoseAgent(max_tests=2)
    }
    results = {}
    for name,agent in agents.items():
        scores=[]
        for _ in range(episodes):
            scores.append(run_episode(agent))
        results[name]=scores

    for name,scores in results.items():
        plt.hist(scores, bins=15, alpha=0.5, label=name)
    plt.xlabel("Total Reward")
    plt.ylabel("Frequency")
    plt.title("Diagnosis Agents Comparison")
    plt.legend()
    plt.show()

def run_episode(agent):
    world = DiagnosisWorld()
    total=0
    while not world.done:
        action = agent.act(world)
        if action[0]=="order_test":
            r,_ = world.order_test(action[1])
        else:
            r,_ = world.diagnose(action[1])
        total+=r
    return total

if __name__=="__main__":
    run_episode_with_logs(RandomDiagnosisAgent(),"Random")
    run_episode_with_logs(GreedyAgent(),"Greedy")
    run_episode_with_logs(TestThenDiagnoseAgent(max_tests=2),"TestThenDiagnose")

    run_experiment(episodes=50)