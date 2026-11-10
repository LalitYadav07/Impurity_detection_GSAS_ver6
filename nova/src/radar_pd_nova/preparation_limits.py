"""One bounded admission limit for independent preparation jobs.

This is Galaxy job concurrency, not a CPU reservation. Scientific worker
pools continue to use the allocation actually reported inside each job.
"""

import os

PREPARATION_JOBS = 8
TRAINING_JOBS = 1
TRAINING_OPERATION = 'preparation-structural-training'
CPU_TRAINING_TOOL_ID = 'neutrons_radar_pd_model_campaign_prototype'
CUDA_TRAINING_TOOL_ID = 'neutrons_radar_pd_preparation_cuda_training_prototype'


def training_device(environ=None):
    """Deployment policy, never a setting imported from a preparation bundle.

    This chooses a producer for new runs. Saved handles and checked optimizer
    continuations retain their original producer even when this policy changes.
    The worker separately verifies its actual device grant before training.
    """
    value = (os.environ if environ is None else environ).get(
        'RADAR_PD_PREPARATION_TRAINING_DEVICE', 'cpu')
    if value not in {'cpu', 'cuda'}:
        raise ValueError('RADAR_PD_PREPARATION_TRAINING_DEVICE must be cpu or cuda')
    return value
