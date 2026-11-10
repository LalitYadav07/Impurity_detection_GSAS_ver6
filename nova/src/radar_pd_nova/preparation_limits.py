"""One bounded admission limit for independent preparation jobs.

This is Galaxy job concurrency, not a CPU reservation. Scientific worker
pools continue to use the allocation actually reported inside each job.
"""

PREPARATION_JOBS = 8
CUDA_TRAINING_TOOL_ID='neutrons_radar_pd_preparation_cuda_training_prototype'
