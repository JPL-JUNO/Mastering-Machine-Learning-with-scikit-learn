import numpy as np
sample = np.random.randint(low=1, high=100, size=10)
print('Original sample: %s' % sample)
print('Sample mean: %s' % sample.mean())

resamples = [np.random.choice(sample, size=sample.shape) for i in range(100)]
print('Number of bootstrap re-samples: %s' % len(resamples))
print('Example re-sample: %s' % resamples[0])

resamplesMean = np.array([resample.mean() for resample in resamples])
print(resamplesMean)
print('Mean of re-samples\' means: %s' % resamplesMean.mean())
