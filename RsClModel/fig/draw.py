import pandas as pd
import matplotlib.pyplot as plot
data=pd.read_excel(r"C:\Users\27997\Desktop\数据汇总\无先验.xlsx")
color=["red","green","yellow","blue","black"]
for i in range(5):
    plot.scatter(data.iloc[i*10:(i+1)*10,0],data.iloc[i*10:(i+1)*10,1])
plot.title("without prior knowledge")
plot.xlabel("error")
plot.ylabel("ratio")
plot.savefig(r"C:\Users\27997\Desktop\无先验.png")

