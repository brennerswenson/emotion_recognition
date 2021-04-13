import subprocess
import random

if __name__ == "__main__":
    batch_size_opts = [128, 64, 32]
    epoch_opts = [50, 100]
    wd_opts = [1e-4, 1e-3]
    orient_opts = [16, 8]
    pix_per_cell_opts = [4, 3]
    dropout_rates = [0.1, 0.25, 0.5]
    lr_opts = [1e-3, 2e-3]

    combos = list()

    # create array with all combinations and then shuffle them
    for wd in wd_opts:
        for e in epoch_opts:
            for batch in batch_size_opts:
                for orient in orient_opts:
                    for pix in pix_per_cell_opts:
                        for dropout in dropout_rates:
                            for lr in lr_opts:
                                tmp_combo = {
                                    "wd": wd,
                                    "epoch": e,
                                    "batch": batch,
                                    "lr": lr,
                                    "dropout": dropout,
                                    "orient": orient,
                                    "pix": pix,
                                    "or": orient,
                                }
                                combos.append(tmp_combo)

    random.shuffle(combos)

    for combo in combos:
        args = f"-b {combo['batch']} -e {combo['e']} -lr {combo['lr']} -o adam -or {combo['or']} -p {combo['pix']} -d {combo['dropout']}"
        subprocess.call(f"venv/Scripts/python src/mlp_train.py {args}")
