All the tests.py train and test an agent over certain data:

test_p.py implements inventory, time, quadratic variation and Price as features and trains the agent over martingale data (data contained in data_martingale)

test_p_drift.py implements inventory, time, quadratic variation and Price as features and trains the agent over martingale with drift data (data contained in data_drift_down_low)

test_p_drift_variable.py implements inventory, time, quadratic variation and Price as features and trains the agent over martingale with two different drifts, -10^{-8} and 10^{-8} (data contained in data_drift_up_down)

test_theoretical_costs gives an estimation of how much we expect to pay with the three different strategies TWAP, AC with gamma>0 and AC with gamma ->0, where gamma is the risk aversion of the agent. 

plot_dati.py simply plots a simulated price

brownian.py and brownian_drift.py are the codes that have been used to simulate the data 
