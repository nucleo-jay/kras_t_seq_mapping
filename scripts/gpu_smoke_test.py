import time
import jax
import jax.numpy as jnp


def main():
    print("JAX:", jax.__version__)
    print("Devices:", jax.devices())

    x = jnp.ones((8000, 8000), dtype=jnp.float32)

    start = time.time()
    y = x @ x
    y.block_until_ready()
    print("First run:", round(time.time() - start, 3), "sec")

    start = time.time()
    y = x @ x
    y.block_until_ready()
    print("Second run:", round(time.time() - start, 3), "sec")


if __name__ == "__main__":
    main()
