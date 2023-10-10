
extract_contr_value() {
    local input_string="$1"
    local value=$(echo "$input_string" | grep "Mean primary value alignment" | grep -o -E '[-]*[0-9]+\.[0-9]*')
    echo "$value"
}

extract_var_value() {
    local input_string="$1"
    local value=$(echo "$input_string" | grep "Permutation Var - mean (over values/traits x perspectives) of var (over perm) (\*10\^3)" | grep -o -E '[-]*[0-9]+\.[0-9]*')
    echo "$value"
}

##### Controllability
#exec > >(tee corr_controllability.csv)

### InstructGPT
## PVQ
pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-instruct-0914_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpti_50_U2`
pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-instruct-0914_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpti_50_U3`
#
## Hofstede
hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-instruct-0914_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_gpti_50_U2`
hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-instruct-0914_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_gpti_50_U3`
#
#
## Big5_50
big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-instruct-0914_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_gpti_50_U2`
big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-instruct-0914_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_gpti_50_U3`
#
echo "GPT-3.5-turbo-instruct , $(extract_contr_value "$pvq_resS2") , $(extract_contr_value "$pvq_resS3") , $(extract_contr_value "$pvq_resU2") , $(extract_contr_value "$pvq_resU3") , $(extract_contr_value "$hof_resS2") , $(extract_contr_value "$hof_resS3") , $(extract_contr_value "$hof_resU2") , $(extract_contr_value "$hof_resU3") , $(extract_contr_value "$big5_resS2") , $(extract_contr_value "$big5_resS3") , $(extract_contr_value "$big5_resU2") , $(extract_contr_value "$big5_resU3")"

exit
###GPT4_50
## PVQ
#pvq_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-4-0314_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt4_50_S2`
pvq_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-4-0314_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt4_50_S3`
#pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-4-0314_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt4_50_U2`
#pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-4-0314_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt4_50_U3`
#
## Hofstede
#hof_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-4-0314_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt4_50_S2`
#hof_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-4-0314_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt4_50_S3`
#hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-4-0314_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt4_50_U2`
hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-4-0314_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt4_50_U3`
#
#
## Big5_50
#big5_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-4-0314_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt4_50_S2`
#big5_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-4-0314_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt4_50_S3`
#big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-4-0314_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt4_50_U2`
big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-4-0314_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt4_50_U3`
#
#echo "GPT-4 , $(extract_contr_value "$pvq_resS2") , $(extract_contr_value "$pvq_resS3") , $(extract_contr_value "$pvq_resU2") , $(extract_contr_value "$pvq_resU3") , $(extract_contr_value "$hof_resS2") , $(extract_contr_value "$hof_resS3") , $(extract_contr_value "$hof_resU2") , $(extract_contr_value "$hof_resU3") , $(extract_contr_value "$big5_resS2") , $(extract_contr_value "$big5_resS3") , $(extract_contr_value "$big5_resU2") , $(extract_contr_value "$big5_resU3")"

### GPT35_50
## PVQ
#pvq_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35_50_S2`
#pvq_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt35_50_S3`
#pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35_50_U2`
#pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt35_50_U3`
#
## Hofstede
#hof_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt35_50_S2`
#hof_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35_50_S3`
#hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt35_50_U2`
#hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35_50_U3`
#
#
## Big5_50
#big5_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35_50_S2`
#big5_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35_50_S3`
#big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35_50_U2`
#big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35_50_U3`
#
#echo "GPT-3.5  , $(extract_contr_value "$pvq_resS2") , $(extract_contr_value "$pvq_resS3") , $(extract_contr_value "$pvq_resU2") , $(extract_contr_value "$pvq_resU3") , $(extract_contr_value "$hof_resS2") , $(extract_contr_value "$hof_resS3") , $(extract_contr_value "$hof_resU2") , $(extract_contr_value "$hof_resU3") , $(extract_contr_value "$big5_resS2") , $(extract_contr_value "$big5_resS3") , $(extract_contr_value "$big5_resU2") , $(extract_contr_value "$big5_resU3")"
#
#
### OpenAssistant
## PVQ
#pvq_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_openassistant_rlhf2_llama30b_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/pvq_oa_50_S2`
#pvq_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_openassistant_rlhf2_llama30b_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/pvq_oa_50_S3`
#pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_openassistant_rlhf2_llama30b_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_oa_50_U2`
#pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_openassistant_rlhf2_llama30b_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_oa_50_U3`
#
#
## Hofstede
#hof_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_openassistant_rlhf2_llama30b_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/hof_oa_50_S2`
#hof_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_openassistant_rlhf2_llama30b_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/hof_oa_50_S3`
#hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_openassistant_rlhf2_llama30b_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_oa_50_U2`
#hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_openassistant_rlhf2_llama30b_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_oa_50_U3`
#
#
## Big5_5
#big5_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_openassistant_rlhf2_llama30b_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/big5_oa_50_S2`
#big5_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_openassistant_rlhf2_llama30b_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/big5_oa_50_S3`
#big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_openassistant_rlhf2_llama30b_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_oa_50_U2`
#big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_openassistant_rlhf2_llama30b_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_oa_50_U3`
#
#echo "OA  , $(extract_contr_value "$pvq_resS2") , $(extract_contr_value "$pvq_resS3") , $(extract_contr_value "$pvq_resU2") , $(extract_contr_value "$pvq_resU3") , $(extract_contr_value "$hof_resS2") , $(extract_contr_value "$hof_resS3") , $(extract_contr_value "$hof_resU2") , $(extract_contr_value "$hof_resU3") , $(extract_contr_value "$big5_resS2") , $(extract_contr_value "$big5_resS3") , $(extract_contr_value "$big5_resU2") , $(extract_contr_value "$big5_resU3") "
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
#echo "StVicuna  , n/a , n/a , $(extract_contr_value "$pvq_resU2") , $(extract_contr_value "$pvq_resU3") , n/a , n/a , $(extract_contr_value "$hof_resU2") , $(extract_contr_value "$hof_resU3") , n/a , n/a , $(extract_contr_value "$big5_resU2") , $(extract_contr_value "$big5_resU3") "
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
#echo "StLM  , $(extract_contr_value "$pvq_resS2") , $(extract_contr_value "$pvq_resS3") , $(extract_contr_value "$pvq_resU2") , $(extract_contr_value "$pvq_resU3") , $(extract_contr_value "$hof_resS2") , $(extract_contr_value "$hof_resS3") , $(extract_contr_value "$hof_resU2") , $(extract_contr_value "$hof_resU3") , $(extract_contr_value "$big5_resS2") , $(extract_contr_value "$big5_resS3") , $(extract_contr_value "$big5_resU2") , $(extract_contr_value "$big5_resU3") "

##### Variance
#exec > >(tee corr_variance.csv)
pvq_resS2=""
pvq_resS3=""
pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_curie_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_U2`
pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_curie_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_U3`

# Hofstede
hof_resS2=""
hof_resS3=""
hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_curie_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt35j_50_U2`
hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_curie_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35j_50_U3`


# Big5_50
big5_resS2=""
big5_resS3=""
big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_curie_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35j_50_U2`
big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_curie_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35j_50_U3`
#
##echo "Curie   & $(extract_var_value "$pvq_resS2") \textbar $(extract_var_value "$pvq_resS3") & $(extract_var_value "$pvq_resU2") \textbar $(extract_var_value "$pvq_resU3") & $(extract_var_value "$hof_resS2") \textbar $(extract_var_value "$hof_resS3") & $(extract_var_value "$hof_resU2") \textbar $(extract_var_value "$hof_resU3") & $(extract_var_value "$big5_resS2") \textbar $(extract_var_value "$big5_resS3") & $(extract_var_value "$big5_resU2") \textbar $(extract_var_value "$big5_resU3") "
#echo "Curie   , $(extract_var_value "$pvq_resS2") , $(extract_var_value "$pvq_resS3") , $(extract_var_value "$pvq_resU2") , $(extract_var_value "$pvq_resU3") , $(extract_var_value "$hof_resS2") , $(extract_var_value "$hof_resS3") , $(extract_var_value "$hof_resU2") , $(extract_var_value "$hof_resU3") , $(extract_var_value "$big5_resS2") , $(extract_var_value "$big5_resS3") , $(extract_var_value "$big5_resU2") , $(extract_var_value "$big5_resU3") "
#
pvq_resS2=""
pvq_resS3=""
pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_babbage_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_U2`
pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_babbage_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_U3`

# Hofstede
hof_resS2=""
hof_resS3=""
hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_babbage_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt35j_50_U2`
hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_babbage_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35j_50_U3`


# Big5_50
big5_resS2=""
big5_resS3=""
big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_babbage_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35j_50_U2`
big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_babbage_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35j_50_U3`
#
##echo "Babbage   & $(extract_var_value "$pvq_resS2") \textbar $(extract_var_value "$pvq_resS3") & $(extract_var_value "$pvq_resU2") \textbar $(extract_var_value "$pvq_resU3") & $(extract_var_value "$hof_resS2") \textbar $(extract_var_value "$hof_resS3") & $(extract_var_value "$hof_resU2") \textbar $(extract_var_value "$hof_resU3") & $(extract_var_value "$big5_resS2") \textbar $(extract_var_value "$big5_resS3") & $(extract_var_value "$big5_resU2") \textbar $(extract_var_value "$big5_resU3") "
#echo "Babbage   , $(extract_var_value "$pvq_resS2") , $(extract_var_value "$pvq_resS3") , $(extract_var_value "$pvq_resU2") , $(extract_var_value "$pvq_resU3") , $(extract_var_value "$hof_resS2") , $(extract_var_value "$hof_resS3") , $(extract_var_value "$hof_resU2") , $(extract_var_value "$hof_resU3") , $(extract_var_value "$big5_resS2") , $(extract_var_value "$big5_resS3") , $(extract_var_value "$big5_resU2") , $(extract_var_value "$big5_resU3") "
#
pvq_resS2=""
pvq_resS3=""
pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_ada_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_U2`
pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_ada_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_U3`

# Hofstede
hof_resS2=""
hof_resS3=""
hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_ada_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt35j_50_U2`
hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_ada_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35j_50_U3`


# Big5_50
big5_resS2=""
big5_resS3=""
big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_ada_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35j_50_U2`
big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_ada_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35j_50_U3`
#
##echo "Ada   & $(extract_var_value "$pvq_resS2") \textbar $(extract_var_value "$pvq_resS3") & $(extract_var_value "$pvq_resU2") \textbar $(extract_var_value "$pvq_resU3") & $(extract_var_value "$hof_resS2") \textbar $(extract_var_value "$hof_resS3") & $(extract_var_value "$hof_resU2") \textbar $(extract_var_value "$hof_resU3") & $(extract_var_value "$big5_resS2") \textbar $(extract_var_value "$big5_resS3") & $(extract_var_value "$big5_resU2") \textbar $(extract_var_value "$big5_resU3") "
#echo "Ada   , $(extract_var_value "$pvq_resS2") , $(extract_var_value "$pvq_resS3") , $(extract_var_value "$pvq_resU2") , $(extract_var_value "$pvq_resU3") , $(extract_var_value "$hof_resS2") , $(extract_var_value "$hof_resS3") , $(extract_var_value "$hof_resU2") , $(extract_var_value "$hof_resU3") , $(extract_var_value "$big5_resS2") , $(extract_var_value "$big5_resS3") , $(extract_var_value "$big5_resU2") , $(extract_var_value "$big5_resU3") "
#
## upstage rp-incite-instruct
pvq_resS2=""
pvq_resS3=""
pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_rp_incite_7b_chat_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_U2`
pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_rp_incite_7b_chat_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_U3`

# Hofstede
hof_resS2=""
hof_resS3=""
hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_rp_incite_7b_chat_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt35j_50_U2`
hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_rp_incite_7b_chat_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35j_50_U3`


# Big5_50
big5_resS2=""
big5_resS3=""
big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_rp_incite_7b_chat_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35j_50_U2`
big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_rp_incite_7b_chat_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35j_50_U3`
#
##echo "Redpaj-incite-chat   & $(extract_var_value "$pvq_resS2") \textbar $(extract_var_value "$pvq_resS3") & $(extract_var_value "$pvq_resU2") \textbar $(extract_var_value "$pvq_resU3") & $(extract_var_value "$hof_resS2") \textbar $(extract_var_value "$hof_resS3") & $(extract_var_value "$hof_resU2") \textbar $(extract_var_value "$hof_resU3") & $(extract_var_value "$big5_resS2") \textbar $(extract_var_value "$big5_resS3") & $(extract_var_value "$big5_resU2") \textbar $(extract_var_value "$big5_resU3") "
#echo "Redpaj-incite-chat   , $(extract_var_value "$pvq_resS2") , $(extract_var_value "$pvq_resS3") , $(extract_var_value "$pvq_resU2") , $(extract_var_value "$pvq_resU3") , $(extract_var_value "$hof_resS2") , $(extract_var_value "$hof_resS3") , $(extract_var_value "$hof_resU2") , $(extract_var_value "$hof_resU3") , $(extract_var_value "$big5_resS2") , $(extract_var_value "$big5_resS3") , $(extract_var_value "$big5_resU2") , $(extract_var_value "$big5_resU3") "
## upstage rp-incite-chat
pvq_resS2=""
pvq_resS3=""
pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_rp_incite_7b_instruct_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_U2`
pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_rp_incite_7b_instruct_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_U3`

# Hofstede
hof_resS2=""
hof_resS3=""
hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_rp_incite_7b_instruct_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt35j_50_U2`
hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_rp_incite_7b_instruct_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35j_50_U3`


# Big5_50
big5_resS2=""
big5_resS3=""
big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_rp_incite_7b_instruct_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35j_50_U2`
big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_rp_incite_7b_instruct_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35j_50_U3`
#
##echo "Redpaj-incite-instruct   & $(extract_var_value "$pvq_resS2") \textbar $(extract_var_value "$pvq_resS3") & $(extract_var_value "$pvq_resU2") \textbar $(extract_var_value "$pvq_resU3") & $(extract_var_value "$hof_resS2") \textbar $(extract_var_value "$hof_resS3") & $(extract_var_value "$hof_resU2") \textbar $(extract_var_value "$hof_resU3") & $(extract_var_value "$big5_resS2") \textbar $(extract_var_value "$big5_resS3") & $(extract_var_value "$big5_resU2") \textbar $(extract_var_value "$big5_resU3") "
#echo "Redpaj-incite-instruct   , $(extract_var_value "$pvq_resS2") , $(extract_var_value "$pvq_resS3") , $(extract_var_value "$pvq_resU2") , $(extract_var_value "$pvq_resU3") , $(extract_var_value "$hof_resS2") , $(extract_var_value "$hof_resS3") , $(extract_var_value "$hof_resU2") , $(extract_var_value "$hof_resU3") , $(extract_var_value "$big5_resS2") , $(extract_var_value "$big5_resS3") , $(extract_var_value "$big5_resU2") , $(extract_var_value "$big5_resU3") "
## llama-65B
pvq_resS2=""
pvq_resS3=""
pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_llama_65B_perm_50__msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_U2`
pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_llama_65B_perm_50__msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_U3`

# Hofstede
hof_resS2=""
hof_resS3=""
hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hof_test_llama_65B_perm_50__msg_2nd_prs/* --save --filename neurips_plots/hof_gpt35j_50_U2`
hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hof_test_llama_65B_perm_50__msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35j_50_U3`


# Big5_50
big5_resS2=""
big5_resS3=""
big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_llama_65B_perm_50__msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35j_50_U2`
big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_llama_65B_perm_50__msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35j_50_U3`
#
##echo "LLaMa-65B   & $(extract_var_value "$pvq_resS2") \textbar $(extract_var_value "$pvq_resS3") & $(extract_var_value "$pvq_resU2") \textbar $(extract_var_value "$pvq_resU3") & $(extract_var_value "$hof_resS2") \textbar $(extract_var_value "$hof_resS3") & $(extract_var_value "$hof_resU2") \textbar $(extract_var_value "$hof_resU3") & $(extract_var_value "$big5_resS2") \textbar $(extract_var_value "$big5_resS3") & $(extract_var_value "$big5_resU2") \textbar $(extract_var_value "$big5_resU3") "
#echo "LLaMa-65B   , $(extract_var_value "$pvq_resS2") , $(extract_var_value "$pvq_resS3") , $(extract_var_value "$pvq_resU2") , $(extract_var_value "$pvq_resU3") , $(extract_var_value "$hof_resS2") , $(extract_var_value "$hof_resS3") , $(extract_var_value "$hof_resU2") , $(extract_var_value "$hof_resU3") , $(extract_var_value "$big5_resS2") , $(extract_var_value "$big5_resS3") , $(extract_var_value "$big5_resU2") , $(extract_var_value "$big5_resU3") "

## upstage llama
#
pvq_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_up_llama_60b_instruct_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/pvq_upllama_50_S2`
pvq_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_up_llama_60b_instruct_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/pvq_upllama_50_S3`
pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_up_llama_60b_instruct_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_upllama_50_U2`
pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_up_llama_60b_instruct_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_upllama_50_U3`

# Hofstede
hof_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_up_llama_60b_instruct_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/hof_upllama_50_S2`
hof_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_up_llama_60b_instruct_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/hof_upllama_50_S3`
hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_up_llama_60b_instruct_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_upllama_50_U2`
hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_up_llama_60b_instruct_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_upllama_50_U3`


# Big5_50
big5_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_up_llama_60b_instruct_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/big5_upllama_50_S2`
big5_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_up_llama_60b_instruct_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/big5_upllama_50_S3`
big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_up_llama_60b_instruct_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_upllama_50_U2`
big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_up_llama_60b_instruct_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_upllama_50_U3`
#
##echo "Upst-LLaMa-66B-instruct   & $(extract_var_value "$pvq_resS2") \textbar $(extract_var_value "$pvq_resS3") & $(extract_var_value "$pvq_resU2") \textbar $(extract_var_value "$pvq_resU3") & $(extract_var_value "$hof_resS2") \textbar $(extract_var_value "$hof_resS3") & $(extract_var_value "$hof_resU2") \textbar $(extract_var_value "$hof_resU3") & $(extract_var_value "$big5_resS2") \textbar $(extract_var_value "$big5_resS3") & $(extract_var_value "$big5_resU2") \textbar $(extract_var_value "$big5_resU3") "
#echo "Upst-LLaMa-66B-instruct   , $(extract_var_value "$pvq_resS2") , $(extract_var_value "$pvq_resS3") , $(extract_var_value "$pvq_resU2") , $(extract_var_value "$pvq_resU3") , $(extract_var_value "$hof_resS2") , $(extract_var_value "$hof_resS3") , $(extract_var_value "$hof_resU2") , $(extract_var_value "$hof_resU3") , $(extract_var_value "$big5_resS2") , $(extract_var_value "$big5_resS3") , $(extract_var_value "$big5_resU2") , $(extract_var_value "$big5_resU3") "
#
## upstage llama2
pvq_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_up_llama2_70b_instruct_v2_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_S2`
pvq_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_up_llama2_70b_instruct_v2_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_S3`
pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_up_llama2_70b_instruct_v2_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_U2`
pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_up_llama2_70b_instruct_v2_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_U3`

# Hofstede
hof_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_up_llama2_70b_instruct_v2_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt35j_50_S2`
hof_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_up_llama2_70b_instruct_v2_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35j_50_S3`
hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_up_llama2_70b_instruct_v2_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt35j_50_U2`
hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_up_llama2_70b_instruct_v2_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35j_50_U3`


# Big5_50
big5_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_up_llama2_70b_instruct_v2_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35j_50_S2`
big5_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_up_llama2_70b_instruct_v2_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35j_50_S3`
big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_up_llama2_70b_instruct_v2_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35j_50_U2`
big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_up_llama2_70b_instruct_v2_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35j_50_U3`
#
##echo "Upst-LLaMa-2-70B-instruct   & $(extract_var_value "$pvq_resS2") \textbar $(extract_var_value "$pvq_resS3") & $(extract_var_value "$pvq_resU2") \textbar $(extract_var_value "$pvq_resU3") & $(extract_var_value "$hof_resS2") \textbar $(extract_var_value "$hof_resS3") & $(extract_var_value "$hof_resU2") \textbar $(extract_var_value "$hof_resU3") & $(extract_var_value "$big5_resS2") \textbar $(extract_var_value "$big5_resS3") & $(extract_var_value "$big5_resU2") \textbar $(extract_var_value "$big5_resU3") "
#echo "Upst-LLaMa-2-70B-instruct   , $(extract_var_value "$pvq_resS2") , $(extract_var_value "$pvq_resS3") , $(extract_var_value "$pvq_resU2") , $(extract_var_value "$pvq_resU3") , $(extract_var_value "$hof_resS2") , $(extract_var_value "$hof_resS3") , $(extract_var_value "$hof_resU2") , $(extract_var_value "$hof_resU3") , $(extract_var_value "$big5_resS2") , $(extract_var_value "$big5_resS3") , $(extract_var_value "$big5_resU2") , $(extract_var_value "$big5_resU3") "
#
### GPT35_50 - june
## PVQ
pvq_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0613_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_S2`
pvq_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0613_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_S3`
pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0613_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_U2`
pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0613_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_U3`

# Hofstede
hof_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0613_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt35j_50_S2`
hof_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0613_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35j_50_S3`
hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0613_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt35j_50_U2`
hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0613_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35j_50_U3`


# Big5_50
big5_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0613_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35j_50_S2`
big5_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0613_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35j_50_S3`
big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0613_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35j_50_U2`
big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0613_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35j_50_U3`
#
##echo "GPT-3.5-0613  & $(extract_var_value "$pvq_resS2") \textbar $(extract_var_value "$pvq_resS3") & $(extract_var_value "$pvq_resU2") \textbar $(extract_var_value "$pvq_resU3") & $(extract_var_value "$hof_resS2") \textbar $(extract_var_value "$hof_resS3") & $(extract_var_value "$hof_resU2") \textbar $(extract_var_value "$hof_resU3") & $(extract_var_value "$big5_resS2") \textbar $(extract_var_value "$big5_resS3") & $(extract_var_value "$big5_resU2") \textbar $(extract_var_value "$big5_resU3") "
#echo "GPT-3.5-0613  , $(extract_var_value "$pvq_resS2") , $(extract_var_value "$pvq_resS3") , $(extract_var_value "$pvq_resU2") , $(extract_var_value "$pvq_resU3") , $(extract_var_value "$hof_resS2") , $(extract_var_value "$hof_resS3") , $(extract_var_value "$hof_resU2") , $(extract_var_value "$hof_resU3") , $(extract_var_value "$big5_resS2") , $(extract_var_value "$big5_resS3") , $(extract_var_value "$big5_resU2") , $(extract_var_value "$big5_resU3") "
#
### GPT35_50
## PVQ
pvq_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35_50_S2`
pvq_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt35_50_S3`
pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35_50_U2`
pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt35_50_U3`

# Hofstede
hof_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt35_50_S2`
hof_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35_50_S3`
hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt35_50_U2`
hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35_50_U3`


# Big5_50
big5_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35_50_S2`
big5_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35_50_S3`
big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35_50_U2`
big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35_50_U3`

#echo "GPT-3.5-0301  , $(extract_var_value "$pvq_resS2") , $(extract_var_value "$pvq_resS3") , $(extract_var_value "$pvq_resU2") , $(extract_var_value "$pvq_resU3") , $(extract_var_value "$hof_resS2") , $(extract_var_value "$hof_resS3") , $(extract_var_value "$hof_resU2") , $(extract_var_value "$hof_resU3") , $(extract_var_value "$big5_resS2") , $(extract_var_value "$big5_resS3") , $(extract_var_value "$big5_resU2") , $(extract_var_value "$big5_resU3") "
#
#
### OpenAssistant
## PVQ
pvq_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_openassistant_rlhf2_llama30b_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/pvq_oa_50_S2`
pvq_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_openassistant_rlhf2_llama30b_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/pvq_oa_50_S3`
pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_openassistant_rlhf2_llama30b_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_oa_50_U2`
pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_openassistant_rlhf2_llama30b_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_oa_50_U3`


# Hofstede
hof_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_openassistant_rlhf2_llama30b_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/hof_oa_50_S2`
hof_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_openassistant_rlhf2_llama30b_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/hof_oa_50_S3`
hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_openassistant_rlhf2_llama30b_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_oa_50_U2`
hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_openassistant_rlhf2_llama30b_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_oa_50_U3`


# Big5_5
big5_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_openassistant_rlhf2_llama30b_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/big5_oa_50_S2`
big5_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_openassistant_rlhf2_llama30b_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/big5_oa_50_S3`
big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_openassistant_rlhf2_llama30b_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_oa_50_U2`
big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_openassistant_rlhf2_llama30b_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_oa_50_U3`
#
#echo "OA  , $(extract_var_value "$pvq_resS2") , $(extract_var_value "$pvq_resS3") , $(extract_var_value "$pvq_resU2") , $(extract_var_value "$pvq_resU3") , $(extract_var_value "$hof_resS2") , $(extract_var_value "$hof_resS3") , $(extract_var_value "$hof_resU2") , $(extract_var_value "$hof_resU3") , $(extract_var_value "$big5_resS2") , $(extract_var_value "$big5_resS3") , $(extract_var_value "$big5_resU2") , $(extract_var_value "$big5_resU3") "
#
#### StableVicuna
## PVQ
pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_stablevicuna_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_sv_50_U2`
pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_stablevicuna_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_sv_50_U3`

# Hofstede
hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_stablevicuna_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_sv_50_U2`
hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_stablevicuna_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_sv_50_U3`

# Big5_5
big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_stablevicuna_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_sv_50_U2`
big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_stablevicuna_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_sv_50_U3`
#
#echo "StVicuna  , n/a , n/a , $(extract_var_value "$pvq_resU2") , $(extract_var_value "$pvq_resU3") , n/a , n/a , $(extract_var_value "$hof_resU2") , $(extract_var_value "$hof_resU3") , n/a , n/a , $(extract_var_value "$big5_resU2") , $(extract_var_value "$big5_resU3") "
#
### StableLM
## PVQ
pvq_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_stablelm_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/pvq_slm_50_S2`
pvq_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_stablelm_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/pvq_slm_50_S3`
pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_stablelm_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_slm_50_U2`
pvq_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_stablelm_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/pvq_slm_50_U3`

# Hofstede
hof_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_stablelm_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/hof_slm_50_S2`
hof_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_stablelm_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/hof_slm_50_S3`
hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_stablelm_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_slm_50_U2`
hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_stablelm_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_slm_50_U3`


# Big5_5
big5_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_stablelm_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/big5_slm_50_S2`
big5_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_stablelm_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/big5_slm_50_S3`
big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_stablelm_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_slm_50_U2`
big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_stablelm_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_slm_50_U3`
#
#echo "StLM  , $(extract_var_value "$pvq_resS2") , $(extract_var_value "$pvq_resS3") , $(extract_var_value "$pvq_resU2") , $(extract_var_value "$pvq_resU3") , $(extract_var_value "$hof_resS2") , $(extract_var_value "$hof_resS3") , $(extract_var_value "$hof_resU2") , $(extract_var_value "$hof_resU3") , $(extract_var_value "$big5_resS2") , $(extract_var_value "$big5_resS3") , $(extract_var_value "$big5_resU2") , $(extract_var_value "$big5_resU3")"
#
