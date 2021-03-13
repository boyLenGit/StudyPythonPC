import matplotlib
import segyio
import matplotlib.pyplot as plt

path1 = 'J:/boyLen_eval_bar_chart.html'
with open(path1, 'r+') as file1:
    string1 = file1.read()
    string1 = string1.replace('precision', 'Col1').replace('Precision', 'Col1')
    file1.close()
with open(path1, 'w') as file2:
    file2.write(string1)
    print(string1)
    file2.close()



