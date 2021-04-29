import subprocess
import random

if __name__ == "__main__":
    # all of the options that are randomly shuffled later
    batch_size_opts = [64, 32]
    epoch_opts = [100, 150]
    wd_opts = [1e-4, 1e-3, 2e-3]
    lr_opts = [1e-3, 2e-3]
    dropout_rates = [0.5, 0.25, 0.125]

    combos = list()

    # create array with all combinations and then shuffle them
    for wd in wd_opts:
        for epoch in epoch_opts:
            for batch in batch_size_opts:
                for lr in lr_opts:
                    for dropout in dropout_rates:
                        tmp_combo = {
                            "wd": wd,
                            "epoch": epoch,
                            "batch": batch,
                            "lr": lr,
                            "dropout": dropout,
                        }
                        combos.append(tmp_combo)

    random.shuffle(combos)

    # call the cnn_train.py module for all of the combinations
    for combo in combos:
        args = f"-b {combo['batch']} -e {combo['epoch']} -lr {combo['lr']} -o adam -wd {combo['wd']} -d {combo['dropout']}"
        subprocess.call(f"venv/Scripts/python src/cnn_train.py {args}")
