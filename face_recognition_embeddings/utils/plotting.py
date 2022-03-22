import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("accuracy_info.csv")

fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(df['Intercept_Limit'],df['Known_Acc'],'b.',label='Known Acc',markersize=3)
plt.plot(df['Intercept_Limit'],df['Unknown_Acc'],'r.',label='Unknown Acc',markersize=3)

plt.ylabel('Percentage')
plt.xlabel('Intercept Limit')
plt.grid()
plt.legend()
plt.savefig('test.png',dpi=300)
