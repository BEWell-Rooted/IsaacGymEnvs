import matplotlib.pyplot as plt
import numpy as np

# Actual data provided by the user
data = {
    'Exponential Decay': ([0.921875, 0.8515625, 0.828125, 0.890625, 0.8125, 0.8671875, 0.859375, 0.8359375, 0.8515625, 0.8425197005271912, 
                         0.8661417365074158, 0.8503937125205994, 0.874015748500824, 0.84375, 0.84375],
                          [0.921875, 0.88671875, 0.8671875, 0.873046875, 0.8609375357627869, 0.8619791865348816, 0.8616071939468384, 
                          0.8583984375, 0.8576388955116272, 0.8561269640922546, 0.8570374250411987, 0.8564836978912354, 
                          0.8578323721885681, 0.8568264842033386, 0.8559547066688538],
                          [5440.89, 5340.63, 5221.81, 5396.10, 5259.56, 5372.92, 5347.08, 5269.22, 5313.99, 5328.48, 5274.25, 5241.74, 5375.86, 5227.84, 5233.84]),
    'Reciprocal': ([0.9212598204612732, 0.8818897604942322, 0.8503937125205994, 0.953125, 0.9344261884689331, 
                         0.9360000491142273, 0.9280000329017639, 0.9280000329017639, 0.9523810148239136, 0.9112902879714966, 
                         0.9112902879714966, 0.9520000219345093, 0.9523810148239136, 0.9112902879714966, 0.9360000491142273],
                   [0.9212598204612732, 0.9015747904777527, 0.8845144510269165, 0.901667058467865, 0.9082188606262207, 
                          0.9128490686416626, 0.9150134921073914, 0.9166367650032043, 0.9206083416938782, 0.9196764826774597, 
                          0.9189141392707825, 0.9216712713241577, 0.9240335822105408, 0.9231233596801758, 0.923981785774231],
                   [5681.13, 5747.23, 5475.32, 5701.13, 5904.45, 5766.80, 5805.53, 5940.96, 5728.27, 5972.41, 5748.34, 5865.96, 5790.68, 5723.46, 5698.65]),
    'Logistic': ([0.8503937125205994, 0.800000011920929, 0.8174603581428528, 0.8188976049423218, 0.7936508655548096, 
                         0.75, 0.796875, 0.7460317611694336, 0.7559055089950562, 0.7680000066757202, 0.7619048357009888, 
                         0.8110235929489136, 0.828125, 0.7716535329818726, 0.7460317611694336],
                 [0.8503937125205994, 0.8251968622207642, 0.8226180076599121, 0.8216879367828369, 0.8160805106163025, 
                          0.8050670623779297, 0.8038967847824097, 0.7966636419296265, 0.7921350002288818, 0.7897214889526367, 
                          0.787192702293396, 0.7891786098480225, 0.7921745181083679, 0.7907087206840515, 0.7877302765846252],
                 [5305.29, 5328.98, 5236.30, 5217.51, 5144.23, 5132.66, 5241.84, 5050.63, 5056.95, 5143.04, 5088.09, 5265.20, 5230.92, 5131.29, 5132.58]),
    'Quadratic': ([0.7843137383460999, 0.7118644118309021, 0.6000000238418579, 0.8103448152542114, 0.7333333492279053, 
                          0.6885245442390442, 0.746666669845581, 0.6307692527770996, 0.6470588445663452, 0.6212121248245239, 
                          0.765625, 0.696969747543335, 0.71875, 0.7285714149475098, 0.6666666865348816],
                  [0.7843137383460999, 0.748089075088501, 0.6987260580062866, 0.726630687713623, 0.7279712557792664, 
                           0.721396803855896, 0.7250068187713623, 0.7132270932197571, 0.7058750987052917, 0.6974087953567505, 
                           0.7036102414131165, 0.7030569314956665, 0.7042641043663025, 0.7060003876686096, 0.7033781409263611],
                  [8128.09, 7416.13, 7581.63, 7506.91, 7469.74, 7336.49, 7239.86, 7021.08, 7015.85, 7197.40, 7318.69, 7019.13, 7199.79, 7342.32, 6546.28]),
    'Tangent Hyperbolic': ([0.9040000438690186, 0.8503937125205994, 0.8253968954086304, 0.9180327653884888, 0.9354838132858276, 
                         0.9112902879714966, 0.9055117964744568, 0.888888955116272, 0.9206349849700928, 0.9200000166893005, 
                         0.9360000491142273, 0.9186991453170776, 0.9268292188644409, 0.9291338324546814, 0.9200000166893005],
                           [0.9040000438690186, 0.8771969079971313, 0.8599302172660828, 0.8744558095932007, 0.8866614699363708, 
                          0.8907662630081177, 0.8928728103637695, 0.8923747539520264, 0.8955147862434387, 0.8979633450508118, 
                          0.9014212489128113, 0.9028610587120056, 0.9047048091888428, 0.9064496755599976, 0.9073530435562134],
                           [5638.61, 5570.63, 5362.86, 5815.89, 5762.50, 5694.72, 5657.81, 5650.26, 5779.85, 5686.67, 5792.96, 5896.66, 5876.65, 5774.23, 5814.42])
}

# Colors for each function
colors = ['#5387DD', '#479A5F', '#DA4C4C', '#EDB732', '#474849']

# X-axis positions
x_pos = np.arange(3)  # Three categories: insertion_success, avg_insert_success, reward
bar_width = 0.15

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Set title with spacing adjustment
plt.title('Test of Non-Linear Reward Scaling Functions in SAPU for Robotic Insertion Task', pad=20)

# Reduce the width of the bars
narrower_bar_width = 0.12

# Plot insertion_success and avg_insert_success on the left Y-axis
ax1.set_ylabel('Success Rate', color='black')
ax1.set_xlabel('Metrics')
ax1.set_ylim(0, 1)  # Set the left Y-axis to start from 0

# Plotting each function's data with unified colors per function and marking the median
for i, (func, (insertion, avg_insertion, reward)) in enumerate(data.items()):
    # Calculate min, max, and median for insertion success and avg insertion success
    ins_min, ins_max = np.min(insertion), np.max(insertion)
    ins_median = np.median(insertion)
    
    avg_min, avg_max = np.min(avg_insertion), np.max(avg_insertion)
    avg_median = np.median(avg_insertion)
    
    # Unified color per function for both insertion and avg insertion success with narrower bars
    ax1.bar(x_pos[0] + i * narrower_bar_width, ins_max - ins_min, bottom=ins_min, width=narrower_bar_width, color=colors[i])
    ax1.plot([x_pos[0] + i * narrower_bar_width - narrower_bar_width/2, x_pos[0] + i * narrower_bar_width + narrower_bar_width/2], 
             [ins_median, ins_median], color='black')
    
    ax1.bar(x_pos[1] + i * narrower_bar_width, avg_max - avg_min, bottom=avg_min, width=narrower_bar_width, color=colors[i], alpha=0.7)
    ax1.plot([x_pos[1] + i * narrower_bar_width - narrower_bar_width/2, x_pos[1] + i * narrower_bar_width + narrower_bar_width/2], 
             [avg_median, avg_median], color='black')

# Plot reward on the right Y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('Reward', color='black')
ax2.set_ylim(0, 22000)  # Set the right Y-axis to start from 0

# Plotting each function's reward data with unified colors and marking the median
for i, (func, (_, _, reward)) in enumerate(data.items()):
    # Calculate min, max, and median for reward
    rew_min, rew_max = np.min(reward), np.max(reward)
    rew_median = np.median(reward)
    
    ax2.bar(x_pos[2] + i * narrower_bar_width, rew_max - rew_min, bottom=rew_min, width=narrower_bar_width, color=colors[i])
    ax2.plot([x_pos[2] + i * narrower_bar_width - narrower_bar_width/2, x_pos[2] + i * narrower_bar_width + narrower_bar_width/2], 
             [rew_median, rew_median], color='black')

# Set x-ticks and labels
ax1.set_xticks(x_pos + (narrower_bar_width * (len(data) - 1)) / 2)
ax1.set_xticklabels(['Insertion Success', 'Avg Insertion Success', 'Reward'])

# Update the legend with function colors, ensuring each legend entry reflects its color
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(data))]
fig.legend(handles, [func for func in data.keys()], loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=5)

# Add a grid and ensure layout fits well
fig.tight_layout()
ax1.grid(True)

# Show plot
plt.show()
