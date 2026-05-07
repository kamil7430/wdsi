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
            evaluation = alg(it, 25)
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

def dyna_q_n_to_iterations():
    plt.clf()

    ns = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    iters = []
    for n in ns:
        print("n:", n)
        for i in range(100):
            print("->", i)
            evalu = algorithms.dyna_q(i, n)
            success_rate = evalu["success_rate"]
            if success_rate >= 0.99:
                iters.append(i)
                break
        else:
            print(":(")

    plt.plot(ns, iters)
    plt.scatter(
        x=ns,
        y=iters,
    )
    plt.grid(True)
    plt.xscale("log")
    plt.xlabel("Value of n")
    plt.ylabel("Iteration count")
    plt.savefig("dyna_q_n_to_iterations.png")

# iteration_count_experiment()
# dyna_q_n_to_iterations()