import math

def polynomial_warmup(step, init_value, end_value, warmup_steps, exponent = 1.0):
    " Polynomial warmup schedule "
    warmup_rate = step / warmup_steps
    lr = init_value + (end_value - init_value) * warmup_rate ** exponent
    return min(end_value, lr)

def polynomial_decay(step, init_value, end_value, decay_steps, exponent = 1.0):
    " Polynomial decay schedule "
    rate = step / decay_steps
    lr = init_value + (end_value - init_value) * rate ** exponent
    return max(end_value, lr)

def cosine_decay(step, init_value, end_value, decay_steps, exponent = 1.0):
    " Cosine decay schedule "
    cosine_decay = 0.5 * (1 + math.cos( math.pi * step / decay_steps))
    return end_value + (init_value - end_value) * cosine_decay ** exponent

decay_schedule_lst = {
    'cosine': cosine_decay,
    'polynomial': polynomial_decay,
}

def warmup_stable_decay(step, init_value, peak_value, min_value, num_steps, warmup_steps, stable_steps = 0, warmup_exponent = 1.0, decay_schedule_name = 'polynomial', decay_exponent = 1.0):
    " Warmup and cosine decay schedule "

    decay_schedule = decay_schedule_lst[decay_schedule_name]
    # warmup phase
    if step < warmup_steps:
        lr_step = polynomial_warmup(
            step = step+1,
            init_value = init_value,
            end_value = peak_value,
            warmup_steps = warmup_steps,
            exponent = warmup_exponent
        )
    # cosine decay phase
    elif step <= warmup_steps + stable_steps:
        lr_step = peak_value
    else:
        lr_step = decay_schedule(
            step = step - warmup_steps - stable_steps + 1,
            init_value = peak_value,
            end_value = min_value,
            decay_steps = num_steps - warmup_steps - stable_steps,
            exponent = decay_exponent
        )
    return lr_step

