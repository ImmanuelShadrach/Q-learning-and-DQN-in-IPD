qlearning.py contains the code for the player implementing a q-learning based strategy  
dqn_pytorch.py contains the code for the player implementing a deep q-network based strategy  
Both players play against a tft (tit-for-tat) player and the results are computed based on points, frequency of cooperation/defection, and continuous length of a certain expected action (always cooperate, always defect, alternate between cooperate and defect).  
functions.py contains the code to calculate the continuous length of actions  
From experimentation with alpha (learning rate) and gamma (discount factor) values, it was determined that in some cases both players performed similarly, while in others the q-learning player fared better than the dqn player.  
Overall, it was concluded that Q-learning outperformed DQN in terms of both convergence speed and behavioral consistency under given training conditions.
