import numpy as np


def turns_and_steps_to_positions(turns, steps):
    assert tuple(turns.shape) == tuple(steps.shape)
    if len(turns.shape) == 1:
        is_batch = False
        turns = turns[np.newaxis]
        steps = steps[np.newaxis]
    elif len(turns.shape) == 2:
        is_batch = True
    else:
        raise ValueError

    batch_size, horizon = turns.shape
    angles = [np.zeros(batch_size)]
    positions = [np.zeros((batch_size, 2))]
    for turn, step in zip(turns.T, steps.T):
        angle = angles[-1] + turn
        position = positions[-1] + step[:, np.newaxis] * np.stack([np.cos(angle), np.sin(angle)], axis=-1)

        angles.append(angle)
        positions.append(position)
    positions = np.stack(positions, axis=1)

    if not is_batch:
        positions = positions[0]

    return positions
