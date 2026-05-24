import time


def compute(limit=1000000):
    start = time.time()

    total = 0
    for i in range(limit):
        total += (i * i) % 97

    elapsed = time.time() - start

    print(f"Iterations: {limit}")
    print(f"Result: {total}")
    print(f"Elapsed: {elapsed:.4f} seconds")


if __name__ == "__main__":
    compute()
