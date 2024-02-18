# %%
from main import run_main

import sys
import traceback

import warnings
warnings.filterwarnings('ignore')

# %%
selected_models     = ['resnet50']
transfer_learnings  = [False] #, True]
n_epochs_s          = [32] #, 50, 150
batch_sizes         = [32] #, 64, 148
lr_s                = [0.001] #, 0.0001, 0,005
wd_s                = [0.005] #, 0.0001, 0,001


continue_list = []

for var_selected_model in selected_models:
    for var_transfer_learning in transfer_learnings:
        for var_n_epochs in n_epochs_s:
            for var_batch_size in batch_sizes:
                for var_lr in lr_s:
                    for var_wd in wd_s:

                        try:

                            variables = [
                                var_selected_model,
                                var_transfer_learning,
                                var_n_epochs,
                                var_batch_size,
                                var_lr,
                                var_wd
                            ]

                            log = [str(x) for x in variables]
                            print(log)
                            if log in continue_list:
                                print('\t--continue--')
                                continue

                            run_main(*variables)

                        except Exception as e:
                            e = str(e)
                            log = f"ERROR ERRROR ERROR\n{e}\n\n"
                            print(log)
                            print(f"traceback -->\n{str(traceback.format_exc())}\n\n")
                            print(f"sys info -->{str(sys.exc_info()[2])}\n\n")


# %%



