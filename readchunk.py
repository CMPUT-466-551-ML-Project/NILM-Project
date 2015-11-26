import pandas as pd

consumption = pd.read_csv('chunk_firsttwelfhours_consumption.csv', index_col=0, header=0);
selected_consumption = pd.concat([consumption['5'],consumption['11'],consumption['13'],consumption['14']], axis=1, join_axes=[consumption.index])
#create dataframe with indicators

print selected_consumption


state=selected_consumption
state[state<=10]=0
state[state>10]=1
sumall=state.sum(axis=1)
sumall=sumall.as_matrix()

print sumall

totalcount=state.count(axis=0)[0]
print totalcount
state=state.as_matrix()
