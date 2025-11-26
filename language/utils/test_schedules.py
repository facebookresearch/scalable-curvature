import schedules as schedules_utils
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


parser = argparse.ArgumentParser(description = 'Experiment parameters')
parser.add_argument('--warmup_steps', type = int, default = 1000)
parser.add_argument('--stable_steps', type = int, default = 8000)
parser.add_argument('--warmup_exponent', type = float, default = 1.0) # exponent for warmup
parser.add_argument('--decay_schedule_name', type = str, default = 'cosine') # decay schedule
parser.add_argument('--decay_exponent', type = float, default = 1.0) # exponent for decay
parser.add_argument('--num_steps', type = int, default = 10_000)
parser.add_argument('--lr_init', type = float, default = 0.0)
parser.add_argument('--lr_peak', type = float, default = 0.1)
parser.add_argument('--lr_min', type = float, default = 0.0)

cfg = parser.parse_args()

# define decay schedule
results = []

for step in range(cfg.num_steps): 
    # warmup
    # step, init_value, peak_value, min_value, num_steps, warmup_steps, stable_steps = 0, warmup_exponent = 1.0, decay_schedule_name = 'polynomial', decay_exponent = 1.0
    lr_step = schedules_utils.warmup_stable_decay(
        step = step,
        init_value = cfg.lr_init,
        peak_value = cfg.lr_peak,
        min_value = cfg.lr_min,
        num_steps = cfg.num_steps,
        warmup_steps = cfg.warmup_steps,
        stable_steps = cfg.stable_steps,
        decay_schedule_name = cfg.decay_schedule_name,
        warmup_exponent = cfg.warmup_exponent,
        decay_exponent = cfg.decay_exponent,
    )
    print(f"Step: {step}, Learning rate: {lr_step}")
    results.append(np.array([step, lr_step]))

# convert to dataframe
df = pd.DataFrame(results, columns = ['step', 'lr'])
# plot the results
fig, ax = plt.subplots(figsize = (10, 7))

ax = sns.lineplot(
    data = df,
    x = 'step',
    y = 'lr',
)

ax.set_title(f"Learning rate schedule: {cfg.decay_schedule_name}")
ax.set_xlabel('Step')
ax.set_ylabel('Learning rate')

fig.savefig(f"plots/lr_schedule_{cfg.decay_schedule_name}_T{cfg.num_steps}_Tw{cfg.warmup_steps}_Ts{cfg.stable_steps}.png", dpi = 300)