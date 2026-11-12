"""
simulation_core.py
==================
Core SimPy process functions shared across notebooks.

Usage:
    from simulation_core import transfer_stage, return_stage, dog_journey, arrival_generator

Parameters (INTAKE_SHAPE, INTAKE_SCALE, etc.) must be defined in the calling scope
before importing these functions, or passed explicitly as arguments.
"""

import numpy as np
import simpy
from scipy import stats


def transfer_stage(env):
    transfer_duration = np.random.exponential(1 / TRANSFER_RATE)
    yield env.timeout(transfer_duration)

def return_stage(env):
    return_duration = np.random.exponential(1 / RETURN_RATE)    
    yield env.timeout(return_duration)

def dog_journey(env, dog_id, results):
    arrival_time = env.now

    # INTAKE STAGE
    intake_duration = stats.gamma.rvs(a=INTAKE_SHAPE, scale=INTAKE_SCALE)
    yield env.timeout(intake_duration)

    # Decision after Intake
    rand = np.random.random()
    if rand < P_INTAKE_TO_ADOPTION:
        # Check for Return after adoption
        if np.random.random() < P_ADOPTION_TO_RETURN:
            yield from return_stage(env)
            # After return - Movement then Adoption
            movement_duration = stats.lognorm.rvs(s=MOVEMENT_SHAPE, scale=MOVEMENT_SCALE)
            yield env.timeout(movement_duration)
        los = env.now - arrival_time
        results.append({'dog_id': dog_id, 'arrival_time': arrival_time, 
                        'los': los, 'outcome': 'Adopted', 'movement': 0})
        return
    elif rand < P_INTAKE_TO_ADOPTION + P_INTAKE_TO_TRANSFER:
        yield from transfer_stage(env)

    # MOVEMEMT STAGE (loop)
    movement_count = 0
    while True:
        movement_duration = stats.lognorm.rvs(s=MOVEMENT_SHAPE, scale=MOVEMENT_SCALE)
        yield env.timeout(movement_duration)
        movement_count += 1

        rand = np.random.random()
        if rand < P_MOVEMENT_TO_ADOPTION / (P_MOVEMENT_TO_ADOPTION + P_MOVEMENT_TO_MOVEMENT + P_MOVEMENT_TO_TRANSFER):
            # Check for Return after adoption
            if np.random.random() < P_ADOPTION_TO_RETURN:
                yield from return_stage(env)
                movement_duration = stats.lognorm.rvs(s=MOVEMENT_SHAPE, scale=MOVEMENT_SCALE)
                yield env.timeout(movement_duration)
                movement_count += 1
            break
        elif rand < (P_MOVEMENT_TO_ADOPTION + P_MOVEMENT_TO_TRANSFER) / (P_MOVEMENT_TO_ADOPTION + P_MOVEMENT_TO_MOVEMENT + P_MOVEMENT_TO_TRANSFER):
            yield from transfer_stage(env)

    # ADOPTION
    los = env.now - arrival_time
    results.append({'dog_id': dog_id, 'arrival_time': arrival_time, 
                    'los': los, 'outcome': 'Adopted', 'movement': movement_count})    


def arrival_generator(env, results, arrival_rate):
    """Generate dog arrivals at exponential inter-arrival times."""
    dog_id = 0
    while True:
        inter_arrival = np.random.exponential(1 / arrival_rate)
        yield env.timeout(inter_arrival)
        dog_id += 1
        env.process(dog_journey(env, dog_id, results))
