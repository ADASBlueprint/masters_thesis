from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

event_acc = EventAccumulator(r"W:\My Documents\McMaster\ThesisProject\Archives\analysis\v1p13")
event_acc.Reload()
# Show all tags in the log file
print(event_acc.Tags())

# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
collision_ratio_step = []
collision_ratio_value = []
for s in event_acc.Scalars('collision_ratio'):
    collision_ratio_step.append(s.step)
    collision_ratio_value.append(s.value)

plt.plot(collision_ratio_step, collision_ratio_value)

plt.xlabel("Episodes")
plt.ylabel("Collision Percentage (%)")

# only one line may be specified; full height
plt.axvline(x=0, color='g', ls='--', lw=1, label='epsilon=0.9')
plt.text(-5,0.15,'epsilon=0.9',rotation=0)


# place legend outside
# plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')

plt.show()

