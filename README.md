# RL-Optimal-Execution

All the tests.py train and test an agent over certain data:

test_p.py implements inventory, time, quadratic variation and Price as features and trains the agent over martingale data 

test_p_drift.py implements inventory, time, quadratic variation and Price as features and trains the agent over martingale with drift data 

test_p_drift_variable.py implements inventory, time, quadratic variation and Price as features and trains the agent over martingale with two different drifts, -10^{-8} and 10^{-8} 

test_theoretical_costs gives an estimation of how much we expect to pay with the three different strategies TWAP, AC with gamma>0 and AC with gamma ->0, where gamma is the risk aversion of the agent. 

plot_dati.py simply plots a simulated price

brownian.py and brownian_drift.py are the codes that have been used to simulate the data 

In order to reproduce our experiments, brownian.py and brownian_drift.py have to be used to simulate 6000 training files and 1000 testing files in the directory /data_X/brownian_train and /data_X/brownian_test where X=martingale, drift_down_low, drift_up_down so that the code can recognise the correct directory. 
