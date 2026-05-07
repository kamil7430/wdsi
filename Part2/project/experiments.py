import matplotlib.pyplot as plt

import algorithms

ALGORITHMS = [algorithms.value_iteration, algorithms.q_learning, algorithms.sarsa, algorithms.dyna_q]

def iteration_count_experiment():
    plt.clf()

    iters = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    for alg in ALGORITHMS:
        print("testing", alg.__name__)
        success_rate = []

        for it in iters:
            print("->", it)
            evaluation = alg(it)
            success_rate.append(evaluation["success_rate"])

        plt.plot(
            iters,
            success_rate,
            label=alg.__name__,
        )

    plt.legend()
    plt.grid(True)
    plt.xscale("log")
    plt.xlabel("Iteration count")
    plt.ylabel("Success rate")
    plt.savefig("iteration_count.png")

# iteration_count_experiment()