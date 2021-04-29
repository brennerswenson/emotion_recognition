import subprocess

import random

if __name__ == "__main__":
    # all of the options that are randomly shuffled later
    kernels = ["rbf", "poly"]
    C_options = [0.05, 0.25, 0.75, 1, 2, 5, 10, 50, 100, 200]
    cluster_factors = [10, 20, 30, 40, 50, 60, 80, 100]
    batch_div_opts = [8, 10, 12, 14, 16]

    combos = list()
    # create array with all combinations and then shuffle them
    for cf in cluster_factors:
        for bd in batch_div_opts:
            for kernel in kernels:
                for c in C_options:
                    tmp_combo = {"cf": cf, "bd": bd, "kernel": kernel, "c": c}
                    combos.append(tmp_combo)

    random.shuffle(combos)

    # call the svc_train.py module for all of the combinations
    for combo in combos:
        args = f"-bd {combo['bd']} -k {combo['kernel']} -c {combo['c']} -cf {combo['cf']}"
        subprocess.call(f"venv/Scripts/python src/svc_train.py {args}")
