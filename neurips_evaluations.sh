
#bash autocrop visualizations/*_gpt35j_50*

#python visualization_scripts/bar_viz.py results_neurips/results_lotr_pvq_test_gpt-4-0314_perm_1_System_msg_2nd_prs/* --save --filename draft/lotr_gpt4
#
#python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-4-0314_perm_1_System_msg_2nd_prs/* --save --filename draft/nat_lang_prof_gpt4
#
#python visualization_scripts/bar_viz.py results_neurips/results_AI_music_expert_pvq_test_gpt-4-0314_perm_1_System_msg_2nd_prs/* --save --filename draft/music_gpt4
#
#python visualization_scripts/bar_viz.py results_neurips/results_hobbies_pvq_test_gpt-4-0314_perm_1_System_msg_2nd_prs/* --save --filename draft/hobbies_gpt4
#

extract_value() {
    local input_string="$1"
    local value=$(echo "$input_string" | grep "Mean primary value alignment" | grep -o -E '[-]*[0-9]+\.[0-9]*')
    echo "$value"
}

extract_var_value() {
    local input_string="$1"
    local value=$(echo "$input_string" | grep "Permutation Var - mean (over values/traits x perspectives) of var (over perm) (\*10\^3)" | grep -o -E '[-]*[0-9]+\.[0-9]*')
    echo "$value"
}


############################3
#### Experiment 3
############################3
### GPT4_5
#echo "\multicolumn{2}{l}{\textit{5 permutations}} \\\\"
## PVQ
#pvq_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-4-0314_perm_5_System_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt4_5_S2`
#pvq_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-4-0314_perm_5_System_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt4_5_S3`
#pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-4-0314_perm_5_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt4_5_U2`
#pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-4-0314_perm_5_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt4_5_U3`
#
## Hofstede
#hof_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-4-0314_perm_5_System_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt4_5_S2`
#hof_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-4-0314_perm_5_System_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt4_5_S3`
#hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-4-0314_perm_5_User_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt4_5_U2`
#hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-4-0314_perm_5_User_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt4_5_U3`
#
#
## Big5_5
#big5_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-4-0314_perm_5_System_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt4_5_S2`
#big5_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-4-0314_perm_5_System_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt4_5_S3`
#big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-4-0314_perm_5_User_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt4_5_U2`
#big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-4-0314_perm_5_User_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt4_5_U3`
#
#echo "GPT-4 & $(extract_value "$pvq_resS2") & $(extract_value "$pvq_resS3") & $(extract_value "$pvq_resU2") & $(extract_value "$pvq_resU3") & $(extract_value "$hof_resS2") & $(extract_value "$hof_resS3") & $(extract_value "$hof_resU2") & $(extract_value "$hof_resU3") & $(extract_value "$big5_resS2") & $(extract_value "$big5_resS3") & $(extract_value "$big5_resU2") & $(extract_value "$big5_resU3") \\\\"

### GPT4_5
#echo "\multicolumn{2}{l}{\textit{5 permutations}} \\\\"
## PVQ
#pvq_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-4-0314_perm_5_System_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt4_5_S2`
#pvq_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-4-0314_perm_5_System_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt4_5_S3`
#pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-4-0314_perm_5_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt4_5_U2`
#pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-4-0314_perm_5_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt4_5_U3`
#
## Hofstede
#hof_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-4-0314_perm_5_System_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt4_5_S2`
#hof_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-4-0314_perm_5_System_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt4_5_S3`
#hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-4-0314_perm_5_User_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt4_5_U2`
#hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-4-0314_perm_5_User_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt4_5_U3`
#
#
## Big5_5
#big5_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-4-0314_perm_5_System_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt4_5_S2`
#big5_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-4-0314_perm_5_System_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt4_5_S3`
#big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-4-0314_perm_5_User_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt4_5_U2`
#big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-4-0314_perm_5_User_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt4_5_U3`
#
#echo "GPT-4 & $(extract_value "$pvq_resS2") & $(extract_value "$pvq_resS3") & $(extract_value "$pvq_resU2") & $(extract_value "$pvq_resU3") & $(extract_value "$hof_resS2") & $(extract_value "$hof_resS3") & $(extract_value "$hof_resU2") & $(extract_value "$hof_resU3") & $(extract_value "$big5_resS2") & $(extract_value "$big5_resS3") & $(extract_value "$big5_resU2") & $(extract_value "$big5_resU3") \\\\"

### GPT35_5
## PVQ
#pvq_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_5_System_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35_5_S2`
#pvq_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_5_System_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt35_5_S3`
#pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_5_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35_5_U2`
#pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_5_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt35_5_U3`
#
## Hofstede
#hof_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_5_System_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt35_5_S2`
#hof_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_5_System_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35_5_S3`
#hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_5_User_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt35_5_U2`
#hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_5_User_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35_5_U3`
#
## Big5_5
#big5_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_5_System_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35_5_S2`
#big5_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_5_System_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35_5_S3`
#big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_5_User_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35_5_U2`
#big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_5_User_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35_5_U3`
#
#echo "GPT-3.5 & $(extract_value "$pvq_resS2") & $(extract_value "$pvq_resS3") & $(extract_value "$pvq_resU2") & $(extract_value "$pvq_resU3") & $(extract_value "$hof_resS2") & $(extract_value "$hof_resS3") & $(extract_value "$hof_resU2") & $(extract_value "$hof_resU3") & $(extract_value "$big5_resS2") & $(extract_value "$big5_resS3") & $(extract_value "$big5_resU2") & $(extract_value "$big5_resU3") \\\\"

## GPT35_5
# PVQ
pvq_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_zephyr-7b-beta_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/pvq_zep_50_S2`
pvq_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_zephyr-7b-beta_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/pvq_zep_50_S3`
pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_zephyr-7b-beta_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_zep_50_U2`
pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_zephyr-7b-beta_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_zep_50_U3`

# Hofstede
hof_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_zephyr-7b-beta_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/hof_zep_50_S2`
hof_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_zephyr-7b-beta_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/hof_zep_50_S3`
hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_zephyr-7b-beta_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_zep_50_U2`
hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_zephyr-7b-beta_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_zep_50_U3`

# Big5_5
big5_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_zephyr-7b-beta_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/big5_zep_50_S2`
big5_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_zephyr-7b-beta_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/big5_zep_50_S3`
big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_zephyr-7b-beta_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_zep_50_U2`
big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_zephyr-7b-beta_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_zep_50_U3`

echo "Zepyr-7B-beta & $(extract_value "$pvq_resS2") & $(extract_value "$pvq_resS3") & $(extract_value "$pvq_resU2") & $(extract_value "$pvq_resU3") & $(extract_value "$hof_resS2") & $(extract_value "$hof_resS3") & $(extract_value "$hof_resU2") & $(extract_value "$hof_resU3") & $(extract_value "$big5_resS2") & $(extract_value "$big5_resS3") & $(extract_value "$big5_resU2") & $(extract_value "$big5_resU3") \\\\"
exit

## GPT_instruct JUNE
# PVQ
pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-instruct-0914_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35in_50_U2`
pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-instruct-0914_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt35in_50_U3`

# Hofstede
hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-instruct-0914_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt35in_50_U2`
hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-instruct-0914_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35in_50_U3`


# Big5_50
big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-instruct-0914_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35in_50_U2`
big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-instruct-0914_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35in_50_U3`

echo "GPT-instruct  & $(extract_value "$pvq_resS2") & $(extract_value "$pvq_resS3") & $(extract_value "$pvq_resU2") & $(extract_value "$pvq_resU3") & $(extract_value "$hof_resS2") & $(extract_value "$hof_resS3") & $(extract_value "$hof_resU2") & $(extract_value "$hof_resU3") & $(extract_value "$big5_resS2") & $(extract_value "$big5_resS3") & $(extract_value "$big5_resU2") & $(extract_value "$big5_resU3") \\\\"

#echo "\multicolumn{2}{l}{\textit{50 permutations}} \\\\"
## GPT35_50 JUNE
# PVQ
pvq_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0613_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_S2`
#pvq_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0613_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_S3`
#pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0613_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_U2`
#pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0613_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_U3`

# Hofstede
#hof_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0613_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt35j_50_S2`
hof_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0613_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35j_50_S3`
#hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0613_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt35j_50_U2`
#hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0613_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35j_50_U3`


# Big5_50
#big5_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0613_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35j_50_S2`
big5_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0613_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35j_50_S3`
#big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0613_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35j_50_U2`
#big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0613_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35j_50_U3`

#echo "GPT-3.5-june  & $(extract_value "$pvq_resS2") & $(extract_value "$pvq_resS3") & $(extract_value "$pvq_resU2") & $(extract_value "$pvq_resU3") & $(extract_value "$hof_resS2") & $(extract_value "$hof_resS3") & $(extract_value "$hof_resU2") & $(extract_value "$hof_resU3") & $(extract_value "$big5_resS2") & $(extract_value "$big5_resS3") & $(extract_value "$big5_resU2") & $(extract_value "$big5_resU3") \\\\"


## GPT35m_50
# PVQ
pvq_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35m_50_S2`
#pvq_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt35m_50_S3`
#pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35m_50_U2`
#pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt35m_50_U3`

# Hofstede
#hof_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt35m_50_S2`
#hof_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35m_50_S3`
hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt35m_50_U2`
#hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35m_50_U3`


# Big5_50
#big5_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35m_50_S2`
#big5_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35m_50_S3`
big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35m_50_U2`
#big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35m_50_U3`

#echo "GPT-3.5  & $(extract_value "$pvq_resS2") & $(extract_value "$pvq_resS3") & $(extract_value "$pvq_resU2") & $(extract_value "$pvq_resU3") & $(extract_value "$hof_resS2") & $(extract_value "$hof_resS3") & $(extract_value "$hof_resU2") & $(extract_value "$hof_resU3") & $(extract_value "$big5_resS2") & $(extract_value "$big5_resS3") & $(extract_value "$big5_resU2") & $(extract_value "$big5_resU3") \\\\"

#
### OpenAssistant
## PVQ
#pvq_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_openassistant_rlhf2_llama30b_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/pvq_oa_50_S2`
#pvq_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_openassistant_rlhf2_llama30b_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/pvq_oa_50_S3`
#pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_openassistant_rlhf2_llama30b_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_oa_50_U2`
#pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_openassistant_rlhf2_llama30b_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_oa_50_U3`
#
## Hofstede
#hof_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_openassistant_rlhf2_llama30b_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/hof_oa_50_S2`
#hof_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_openassistant_rlhf2_llama30b_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/hof_oa_50_S3`
#hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_openassistant_rlhf2_llama30b_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_oa_50_U2`
#hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_openassistant_rlhf2_llama30b_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_oa_50_U3`
#
## Big5_5
#big5_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_openassistant_rlhf2_llama30b_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/big5_oa_50_S2`
#big5_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_openassistant_rlhf2_llama30b_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/big5_oa_50_S3`
#big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_openassistant_rlhf2_llama30b_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_oa_50_U2`
#big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_openassistant_rlhf2_llama30b_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_oa_50_U3`
#
#echo "OA  & $(extract_value "$pvq_resS2") & $(extract_value "$pvq_resS3") & $(extract_value "$pvq_resU2") & $(extract_value "$pvq_resU3") & $(extract_value "$hof_resS2") & $(extract_value "$hof_resS3") & $(extract_value "$hof_resU2") & $(extract_value "$hof_resU3") & $(extract_value "$big5_resS2") & $(extract_value "$big5_resS3") & $(extract_value "$big5_resU2") & $(extract_value "$big5_resU3") \\\\"
#
#### StableVicuna
## PVQ
#pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_stablevicuna_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_sv_50_U2`
#pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_stablevicuna_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_sv_50_U3`
#
## Hofstede
#hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_stablevicuna_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_sv_50_U2`
#hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_stablevicuna_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_sv_50_U3`
#
## Big5_5
#big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_stablevicuna_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_sv_50_U2`
#big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_stablevicuna_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_sv_50_U3`
#
#echo "StVicuna  & n/a & n/a & $(extract_value "$pvq_resU2") & $(extract_value "$pvq_resU3") & n/a & n/a & $(extract_value "$hof_resU2") & $(extract_value "$hof_resU3") & n/a & n/a & $(extract_value "$big5_resU2") & $(extract_value "$big5_resU3") \\\\"
#
### StableLM
## PVQ
#pvq_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_stablelm_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/pvq_slm_50_S2`
#pvq_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_stablelm_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/pvq_slm_50_S3`
#pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_stablelm_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_slm_50_U2`
#pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_stablelm_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_slm_50_U3`
#
## Hofstede
#hof_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_stablelm_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/hof_slm_50_S2`
#hof_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_stablelm_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/hof_slm_50_S3`
#hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_stablelm_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_slm_50_U2`
#hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_stablelm_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_slm_50_U3`
#
#
## Big5_5
#big5_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_stablelm_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/big5_slm_50_S2`
#big5_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_stablelm_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/big5_slm_50_S3`
#big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_stablelm_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_slm_50_U2`
#big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_stablelm_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_slm_50_U3`
#
#echo "StLM  & $(extract_value "$pvq_resS2") & $(extract_value "$pvq_resS3") & $(extract_value "$pvq_resU2") & $(extract_value "$pvq_resU3") & $(extract_value "$hof_resS2") & $(extract_value "$hof_resS3") & $(extract_value "$hof_resU2") & $(extract_value "$hof_resU3") & $(extract_value "$big5_resS2") & $(extract_value "$big5_resS3") & $(extract_value "$big5_resU2") & $(extract_value "$big5_resU3") \\\\"
#
#### LLaMa-65B
## PVQ
#pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_llama_65B_perm_50__msg_2nd_prs/* --save --filename neurips_plots/pvq_llama_50_U2`
#pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_llama_65B_perm_50__msg_3rd_prs/* --save --filename neurips_plots/pvq_llama_50_U3`
#
## Hofstede
#hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hof_test_llama_65B_perm_50__msg_2nd_prs/* --save --filename neurips_plots/hof_llama_50_U2`
#hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hof_test_llama_65B_perm_50__msg_3rd_prs/* --save --filename neurips_plots/hof_llama_50_U3`
#
## Big5_5
#big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_llama_65B_perm_50__msg_2nd_prs/* --save --filename neurips_plots/big5_llama_50_U2`
#big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_llama_65B_perm_50__msg_3rd_prs/* --save --filename neurips_plots/big5_llama_50_U3`
#
#echo "LLaMa-65B  & n/a & n/a & $(extract_value "$pvq_resU2") & $(extract_value "$pvq_resU3") & n/a & n/a & $(extract_value "$hof_resU2") & $(extract_value "$hof_resU3") & n/a & n/a & $(extract_value "$big5_resU2") & $(extract_value "$big5_resU3") \\\\"



##############################3
###### Experiment 4
##############################3
#
#echo "\multicolumn{2}{l}{\textit{50 permutations}} \\\\"
### GPT35_50
## PVQ
#pvq_m=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_50_System_msg_2nd_prs_intensity_slight/* --save --filename neurips_plots/pvq_gpt35_S2`
#pvq_h=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_50_System_msg_2nd_prs_intensity_more/* --save --filename neurips_plots/pvq_gpt35_S2`
#pvq_eh=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35_S2`
#
## Hofstede
#hof_m=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_50_User_msg_2nd_prs_intensity_slight/* --save --filename neurips_plots/hof_gpt35_U2`
#hof_h=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_50_User_msg_2nd_prs_intensity_more/* --save --filename neurips_plots/hof_gpt35_U2`
#hof_eh=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt35_U2`
#
## Big5_50
#big5_m=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_50_User_msg_2nd_prs_intensity_slight/* --save --filename neurips_plots/big5_gpt35_U2`
#big5_h=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_50_User_msg_2nd_prs_intensity_more/* --save --filename neurips_plots/big5_gpt35_U2`
#big5_eh=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35_U2`
#
#echo "GPT-3.5 & $(extract_value "$pvq_m") & $(extract_value "$pvq_h") & $(extract_value "$pvq_eh") & $(extract_value "$hof_m") & $(extract_value "$hof_h") & $(extract_value "$hof_eh") & $(extract_value "$big5_m") & $(extract_value "$big5_h") & $(extract_value "$big5_eh") \\\\"
#
### OpenAssistant
## PVQ
#pvq_m=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_openassistant_rlhf2_llama30b_perm_50_User_msg_2nd_prs_intensity_slight/* --save --filename neurips_plots/pvq_gpt35_U2`
#pvq_h=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_openassistant_rlhf2_llama30b_perm_50_User_msg_2nd_prs_intensity_more/* --save --filename neurips_plots/pvq_gpt35_U2`
#pvq_eh=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_openassistant_rlhf2_llama30b_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35_U2`
#
## Hofstede
#hof_m=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_openassistant_rlhf2_llama30b_perm_50_User_msg_3rd_prs_intensity_slight/* --save --filename neurips_plots/hof_gpt35_U3`
#hof_h=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_openassistant_rlhf2_llama30b_perm_50_User_msg_3rd_prs_intensity_more/* --save --filename neurips_plots/hof_gpt35_U3`
#hof_eh=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_openassistant_rlhf2_llama30b_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35_U3`
#
## Big5_5
#big5_m=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_openassistant_rlhf2_llama30b_perm_50_User_msg_3rd_prs_intensity_slight/* --save --filename neurips_plots/big5_gpt35_U3`
#big5_h=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_openassistant_rlhf2_llama30b_perm_50_User_msg_3rd_prs_intensity_more/* --save --filename neurips_plots/big5_gpt35_U3`
#big5_eh=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_openassistant_rlhf2_llama30b_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35_U3`
#
#echo "OA & $(extract_value "$pvq_m") & $(extract_value "$pvq_h") & $(extract_value "$pvq_eh") & $(extract_value "$hof_m") & $(extract_value "$hof_h") & $(extract_value "$hof_eh") & $(extract_value "$big5_m") & $(extract_value "$big5_h") & $(extract_value "$big5_eh") \\\\"
#
#### StableVicuna
## PVQ
#pvq_m=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_stablevicuna_perm_50_User_msg_2nd_prs_intensity_slight/* --save --filename neurips_plots/pvq_gpt35_U2`
#pvq_h=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_stablevicuna_perm_50_User_msg_2nd_prs_intensity_more/* --save --filename neurips_plots/pvq_gpt35_U2`
#pvq_eh=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_stablevicuna_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35_U2`
#
## Hofstede
#hof_m=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_stablevicuna_perm_50_User_msg_3rd_prs_intensity_slight/* --save --filename neurips_plots/hof_gpt35_U3`
#hof_h=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_stablevicuna_perm_50_User_msg_3rd_prs_intensity_more/* --save --filename neurips_plots/hof_gpt35_U3`
#hof_eh=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_stablevicuna_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35_U3`
#
## Big5_5
#big5_m=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_stablevicuna_perm_50_User_msg_3rd_prs_intensity_slight/* --save --filename neurips_plots/big5_gpt35_U3`
#big5_h=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_stablevicuna_perm_50_User_msg_3rd_prs_intensity_more/* --save --filename neurips_plots/big5_gpt35_U3`
#big5_eh=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_stablevicuna_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35_U3`
#
#echo "StVicuna & $(extract_value "$pvq_m") & $(extract_value "$pvq_h") & $(extract_value "$pvq_eh") & $(extract_value "$hof_m") & $(extract_value "$hof_h") & $(extract_value "$hof_eh") & $(extract_value "$big5_m") & $(extract_value "$big5_h") & $(extract_value "$big5_eh") \\\\"
#
### StableLM
## PVQ
#pvq_m=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_stablelm_perm_50_User_msg_2nd_prs_intensity_slight/* --save --filename neurips_plots/pvq_gpt35_U2`
#pvq_h=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_stablelm_perm_50_User_msg_2nd_prs_intensity_more/* --save --filename neurips_plots/pvq_gpt35_U2`
#pvq_eh=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_stablelm_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35_U2`
#
## Hofstede
#hof_m=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_stablelm_perm_50_System_msg_3rd_prs_intensity_slight/* --save --filename neurips_plots/hof_gpt35_S3`
#hof_h=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_stablelm_perm_50_System_msg_3rd_prs_intensity_more/* --save --filename neurips_plots/hof_gpt35_S3`
#hof_eh=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_stablelm_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35_S3`
#
## Big5_5
#big5_m=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_stablelm_perm_50_User_msg_2nd_prs_intensity_slight/* --save --filename neurips_plots/big5_gpt35_U2`
#big5_h=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_stablelm_perm_50_User_msg_2nd_prs_intensity_more/* --save --filename neurips_plots/big5_gpt35_U2`
#big5_eh=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_stablelm_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35_U2`
#
#echo "StLM & $(extract_value "$pvq_m") & $(extract_value "$pvq_h") & $(extract_value "$pvq_eh") & $(extract_value "$hof_m") & $(extract_value "$hof_h") & $(extract_value "$hof_eh") & $(extract_value "$big5_m") & $(extract_value "$big5_h") & $(extract_value "$big5_eh") \\\\"
