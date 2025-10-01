# PEAS Lab — Week 3

This project implements **PEAS (Performance, Environment, Actuators, Sensors) agents** in two classic AI domains:  

1. **Vacuum World** — an agent that cleans a grid with dirt respawn.  
   - Agents: Random, Local Reflex, Full Reflex.  
   - Experiments: effect of dirt spawn probability on mean reward.  
   - Outputs: step-by-step logs + performance plots.  

2. **Medical Diagnosis** — an agent diagnosing between flu, cold, or allergy using noisy tests.  
   - Agents: Random, Greedy, Test-then-Diagnose (Bayesian).  
   - Experiments: comparing rewards and trade-offs between test cost and accuracy.  
   - Outputs: posterior updates, step-by-step logs, reward histograms.  

The lab explores how **PEAS choices, observability, and stochasticity** affect agent performance.  
