import os
import numpy as np
import pandas as pd

# output_dir="/work/r11921042/output/rl_agent"
output_dir="/home/dmnph/reborn_output/rl_agent"

# coef_ter_list=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
# coef_len_list=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]

# coef_ter_list=["0.0", "0.2"]
# coef_len_list=["0.0", "0.2"]

coef_ter_list=["0.2"]
coef_len_list=["0.2"]


# posttag_list = ["_LMnosil_Tnosil", "_LMnosil_Tsil", "_LMsil_Tnosil", "_LMsil_Tsil"]
posttag_list = ["_LMsil_Tsil"]
seed_list=['3', '11', '13', '17', '113', '117']

# posttag_list = ["_LMnosil_Tnosil"]
# seed_list=['3']

lr="1e-4"
epoch="40"

# prefix="timit_matched"
# prefix="mls_es/MLS_ES"
# prefix="mls_de/MLS_DE"
# prefix="mls_pt/MLS_PT"
# prefix="mls_nl/MLS_NL"
# prefix="mls_fr/MLS_FR"
# prefix="mls_it/MLS_IT"
# prefix="ls_en/hb_LS_EN"
prefix="ls_en/LS_EN"
# postfix="_postITER2"
postfix=""

# list of ckpt_type
# ckpt_typeS = ["best", "epoch40"]
ckpt_typeS = ["best"]

# list of testing set
# test_setS = ["test", "valid", "all-test"]

## For LibriSpeech
test_setS = ["valid", "dev-other", "test", "test-other"]
# test_setS = ["test"]

# Get the output_list from traverse the output directory and filter the directory name with prefix and postfix
# Sort the output_list by the directory name
# output_list = [d for d in os.listdir(output_dir) if d.startswith(prefix) and d.endswith(postfix)]
# output_list.sort()

# For each output directory, get the data from the result csv file and store them into a dataframe

df = pd.DataFrame()

# for output in output_list:

for seed in seed_list:
    for posttag in posttag_list:
        for coef_ter in coef_ter_list:
            for coef_len in coef_len_list:
                # timit_matched_pplNorm1.0_tokerr0.0_lenratio0.6_lr1e-4_epoch500_seed3
                output = f"{prefix}_pplNorm1.0_tokerr{coef_ter}_lenratio{coef_len}_lr{lr}_epoch{epoch}_seed{seed}{posttag}{postfix}"
                print(f"output: {output}")

                # Get the data from the val_scores csv file
                val_scores_csv = os.path.join(output_dir, output, "val_scores.csv")

                try:
                    val_scores = pd.read_csv(val_scores_csv)
                    # Change column "Unnamed: 0" to "run_name"
                    if "Unnamed: 0" in val_scores.columns:
                        val_scores.rename(columns={"Unnamed: 0": "run_name"}, inplace=True)
                    
                    # # backup the file to prevent data loss
                    # import shutil
                    # shutil.copy(val_scores_csv, val_scores_csv.replace(".csv", "_backup.csv"))
                    # val_scores.to_csv(val_scores_csv, encoding="utf-8", index=False)
                except:
                    print(f"val_scores_csv: {val_scores_csv} does not exist")
                    continue


                for ckpt_type in ckpt_typeS:
                    # Check if the ckpt_type is in the val_scores dataframe
                    if not val_scores["model_name"].str.contains(ckpt_type).any():
                        if ckpt_type == "best":
                            # select the max eval_score row as the best row, make it format as val_scores["model_name"].str.contains(ckpt_type)
                            row = val_scores["eval_score"] == val_scores["eval_score"].max()

                        else:
                            print(f"ckpt_type: {ckpt_type} does not exist in the val_scores dataframe")

                            data_row = {
                                "run_name": val_scores["run_name"].values[-1],
                                "posttag": posttag, 
                                "ckpt_type": ckpt_type,
                                "coef_ter": coef_ter,
                                "coef_len": coef_len,
                            }

                            df = df.append(data_row, ignore_index=True)
                            continue

                    else:
                        row = val_scores["model_name"].str.contains(ckpt_type)

                    print(row)

                    # Store the data into a dictionary
                    data_row = {}
                    for col in val_scores.columns:
                        data_row[col] = val_scores[row][col].values[-1]

                        if col == "model_name":
                            data_row["posttag"] = posttag
                            data_row["ckpt_type"] = ckpt_type
                            data_row["coef_ter"] = coef_ter
                            data_row["coef_len"] = coef_len
                    
                    
                    # Add columns of each test_set
                    for test_set in test_setS:
                        data_row["{}_PER".format(test_set)] = np.nan
                        # data_row["{}_lenient_f1".format(test_set)] = np.nan
                        # data_row["{}_harsh_f1".format(test_set)] = np.nan

                    # For each ckpt_type, get the data from the result csv file 
                    for test_set in test_setS:
                        test_scores_txt = os.path.join(output_dir, output, "results/result_{}_{}.txt".format(test_set, ckpt_type))

                        # Check if the test_scores_txt exists
                        if not os.path.exists(test_scores_txt):
                            continue

                        # Get PER, lenient_f1, harsh_f1 from the result txt file and store them into the dataframe
                        with open(test_scores_txt, "r") as f:
                            lines = f.readlines()
                            for line in lines:
                                # Record every line
                                if ':' in line:
                                    data_row[f"{test_set}_{line.split(':')[0]}"] = line.split(':')[1].split()[0].strip()
                                # if line.startswith("PER"):
                                #     PER = float(line.split()[1])
                                # elif line.startswith("f1"):
                                #     lenient_f1 = float(line.split()[-2])
                                #     harsh_f1 = float(line.split()[-1])
                        # data_row["{}_PER".format(test_set)] = PER   

                        # if test_set == "test":
                        #     data_row["{}_lenient_f1".format(test_set)] = lenient_f1
                        #     data_row["{}_harsh_f1".format(test_set)] = harsh_f1

                    # Add data to the dataframe
                    df = df.append(data_row, ignore_index=True)

# Save the dataframe to csv file
df.to_csv(f"./result_{prefix.split('/')[0]}.csv", index=False)