# DENSE_Machine_Learning

This is a brief data analysis project I did under the supervision of Professor John Stratton, relating to his DENSE software for simulating delay differential equations. The goal was to see if the performance of a simulation, measured in runtime intervals, could be predicted based on the rates and concentrations of each variable in the simulation at the beginning of each time interval. I found that for one particular simulation (her_model_2014), linear regression could predict the performance of the simulation to within about a third of a standard deviation.

The original program was written using JupyterLab, which is why comments like # In[14]: appear as a result of converting to .py file. The source code for DENSE and the her_2014 model can be found here: https://github.com/johnastratton/DENSE.
