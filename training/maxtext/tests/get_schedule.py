import jax.numpy as jnp
import optax

# 这里粘贴你的 create_learning_rate_schedule
def create_learning_rate_schedule(config):
    """Creates a warmup and cosine decay learning rate schedule"""
    def make_cos_schedule(init_lr, final_lr, len_steps):
        def schedule(step):
            pct = (step) / len_steps
            a = 0.5 * (jnp.cos(jnp.pi * pct) + 1)
            lr = init_lr * a + final_lr * (1 - a)
            return lr
        return schedule

    lr = config.learning_rate
    cos_final_lr = lr * config.cosine_learning_rate_final_fraction

    warmup_steps = int(config.learning_rate_schedule_steps * config.warmup_steps_fraction)
    cos_steps = config.learning_rate_schedule_steps - warmup_steps
    constant_zero_steps = config.steps - config.learning_rate_schedule_steps

    warmup_schedule = optax.linear_schedule(init_value=0.0, end_value=lr, transition_steps=warmup_steps)
    cos_schedule = make_cos_schedule(lr, cos_final_lr, cos_steps)
    constant_schedule = optax.constant_schedule(0.0)

    pieces = [warmup_schedule, cos_schedule]
    boundaries = [
        warmup_steps,
        warmup_steps + cos_steps,
    ]

    if constant_zero_steps > 0:
        pieces.append(constant_schedule)
        boundaries.append(warmup_steps + cos_steps + constant_zero_steps)

    return optax.join_schedules(pieces, boundaries)


# ----------------------------
# Example config
# ----------------------------
class Config:
    learning_rate = 3e-4
    cosine_learning_rate_final_fraction = 0.1
    warmup_steps_fraction = 0.01
    learning_rate_schedule_steps = 50000
    steps = 50000   # include constant zero section


def main():
    config = Config()
    schedule_fn = create_learning_rate_schedule(config)

    # 生成 schedule
    lrs = [float(schedule_fn(step)) for step in range(config.steps)]

    # 写入文件
    out_path = "lr_schedule.txt"
    with open(out_path, "w") as f:
        for lr in lrs:
            f.write(f"{lr:.8e}\n")

    print(f"[INFO] Wrote {len(lrs)} learning rate values to {out_path}")


if __name__ == "__main__":
    main()