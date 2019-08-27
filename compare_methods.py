import matplotlib.pyplot as plt
import pandas as pd

plot_name = 'Stats'


def plot_comp(output_dir, ax):
    data = pd.read_csv(output_dir + '/progress.txt', sep="\t")
    data.index = data['TotalEnvInteracts']
    data_plot = data[['AverageEpRet', 'EpLen']]
    cols = data_plot.columns
    prefix = output_dir.split('/')[-1] + ' '
    print('Adding data for ' + prefix)
    data_plot.columns = [prefix + entry for entry in cols]
    data_plot = data_plot.rolling(5).mean()
    right_data = [prefix + entry for entry in cols[1:]]
    data_plot.plot(secondary_y=right_data, ax=ax)


output_dirs = ['logging/new_environment/td3', 'logging/new_environment/naf']

fig, ax = plt.subplots()
for output_dir in output_dirs:
    plot_comp(output_dir=output_dir, ax=ax)

ax.set_title('name')
fig.show()
