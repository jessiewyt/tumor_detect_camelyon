import numpy as np

# Downsample negative slides
def downsampler(all_samples):
  idx=all_samples.index[all_samples['is_tumor'] == False].tolist()
  num_to_drop = sum(all_samples['is_tumor'] == False) - sum(all_samples['is_tumor'] == True)
  drop_indices = np.random.choice(idx, num_to_drop, replace=False)
  # print(len(drop_indices))
  # print(all_samples.shape)
  balanced_samples = all_samples.drop(drop_indices)
  # print(balanced_samples.shape)
  # reorder the index.
  balanced_samples.reset_index(drop=True, inplace=True)
  return balanced_samples