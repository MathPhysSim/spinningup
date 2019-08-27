import matplotlib.pyplot as plt
import pandas as pd

plot_name = 'Stats'

fig, ax = plt.subplots()
def plot_comp(output_dir, ax):
    data = pd.read_csv(output_dir+'/progress.txt', sep="\t")
    data.index = data['TotalEnvInteracts']
    data_plot= data[[ 'AverageEpRet']]
    cols = data_plot.columns
    prefix = output_dir.split('/')[-1] + ' '
    data_plot.columns = [prefix + entry for entry in cols]
    right_data = [prefix + entry for entry in cols]
    data_plot.plot(secondary_y=right_data,ax=ax)


# output_dir = 'logging/DDPG'
# plot_comp(output_dir=output_dir, ax=ax)
#
# output_dir = 'logging/TD3'
# plot_comp(output_dir=output_dir, ax=ax)

output_dir = 'logging/new_environment/td3/'
plot_comp(output_dir=output_dir, ax=ax)

ax.set_title('name')
fig.show()