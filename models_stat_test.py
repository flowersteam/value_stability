#! python3

from pathlib import Path
import json
from collections import defaultdict
import scipy.stats as stats
from termcolor import colored

data=defaultdict(dict)
# use t-tests to compare

## Zephyr
data["zephyr"]["pvq_resS2"] = "results_neurips/results_nat_lang_prof_pvq_test_zephyr-7b-beta_perm_50_System_msg_2nd_prs/"
data["zephyr"]["pvq_resS3"] = "results_neurips/results_nat_lang_prof_pvq_test_zephyr-7b-beta_perm_50_System_msg_3rd_prs/"
data["zephyr"]["pvq_resU2"] = "results_neurips/results_nat_lang_prof_pvq_test_zephyr-7b-beta_perm_50_User_msg_2nd_prs/"
data["zephyr"]["pvq_resU3"] = "results_neurips/results_nat_lang_prof_pvq_test_zephyr-7b-beta_perm_50_User_msg_3rd_prs/"
data["zephyr"]["hof_resS2"] = "results_neurips/results_nat_lang_prof_hofstede_test_zephyr-7b-beta_perm_50_System_msg_2nd_prs/"
data["zephyr"]["hof_resS3"] = "results_neurips/results_nat_lang_prof_hofstede_test_zephyr-7b-beta_perm_50_System_msg_3rd_prs/"
data["zephyr"]["hof_resU2"] = "results_neurips/results_nat_lang_prof_hofstede_test_zephyr-7b-beta_perm_50_User_msg_2nd_prs/"
data["zephyr"]["hof_resU3"] = "results_neurips/results_nat_lang_prof_hofstede_test_zephyr-7b-beta_perm_50_User_msg_3rd_prs/"
data["zephyr"]["big5_resS2"] = "results_neurips/results_nat_lang_prof_big5_test_zephyr-7b-beta_perm_50_System_msg_2nd_prs/"
data["zephyr"]["big5_resS3"] = "results_neurips/results_nat_lang_prof_big5_test_zephyr-7b-beta_perm_50_System_msg_3rd_prs/"
data["zephyr"]["big5_resU2"] = "results_neurips/results_nat_lang_prof_big5_test_zephyr-7b-beta_perm_50_User_msg_2nd_prs/"
data["zephyr"]["big5_resU3"] = "results_neurips/results_nat_lang_prof_big5_test_zephyr-7b-beta_perm_50_User_msg_3rd_prs/"

## GPT4
# data["gpt4"]["pvq_resS2"] = "results_neurips/results_nat_lang_prof_pvq_test_gpt-4-0314_perm_50_System_msg_2nd_prs/"
data["gpt4"]["pvq_resS3"] = "results_neurips/results_nat_lang_prof_pvq_test_gpt-4-0314_perm_50_System_msg_3rd_prs/"
# data["gpt4"]["pvq_resU2"] = "results_neurips/results_nat_lang_prof_pvq_test_gpt-4-0314_perm_50_User_msg_2nd_prs/"
# data["gpt4"]["pvq_resU3"] = "results_neurips/results_nat_lang_prof_pvq_test_gpt-4-0314_perm_50_User_msg_3rd_prs/"
# data["gpt4"]["hof_resS2"] = "results_neurips/results_nat_lang_prof_hofstede_test_gpt-4-0314_perm_50_System_msg_2nd_prs/"
# data["gpt4"]["hof_resS3"] = "results_neurips/results_nat_lang_prof_hofstede_test_gpt-4-0314_perm_50_System_msg_3rd_prs/"
# data["gpt4"]["hof_resU2"] = "results_neurips/results_nat_lang_prof_hofstede_test_gpt-4-0314_perm_50_User_msg_2nd_prs/"
data["gpt4"]["hof_resU3"] = "results_neurips/results_nat_lang_prof_hofstede_test_gpt-4-0314_perm_50_User_msg_3rd_prs/"
# data["gpt4"]["big5_resS2"] = "results_neurips/results_nat_lang_prof_big5_test_gpt-4-0314_perm_50_System_msg_2nd_prs/"
# data["gpt4"]["big5_resS3"] = "results_neurips/results_nat_lang_prof_big5_test_gpt-4-0314_perm_50_System_msg_3rd_prs/"
# data["gpt4"]["big5_resU2"] = "results_neurips/results_nat_lang_prof_big5_test_gpt-4-0314_perm_50_User_msg_2nd_prs/"
data["gpt4"]["big5_resU3"] = "results_neurips/results_nat_lang_prof_big5_test_gpt-4-0314_perm_50_User_msg_3rd_prs/"

## GPT35m
data["gpt35m"]["pvq_resS2"] = "results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_50_System_msg_2nd_prs/"
data["gpt35m"]["pvq_resS3"] = "results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_50_System_msg_3rd_prs/"
data["gpt35m"]["pvq_resU2"] = "results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_50_User_msg_2nd_prs/"
data["gpt35m"]["pvq_resU3"] = "results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_50_User_msg_3rd_prs/"
data["gpt35m"]["hof_resS2"] = "results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_50_System_msg_2nd_prs/"
data["gpt35m"]["hof_resS3"] = "results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_50_System_msg_3rd_prs/"
data["gpt35m"]["hof_resU2"] = "results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_50_User_msg_2nd_prs/"
data["gpt35m"]["hof_resU3"] = "results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_50_User_msg_3rd_prs/"
data["gpt35m"]["big5_resS2"] = "results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_50_System_msg_2nd_prs/"
data["gpt35m"]["big5_resS3"] = "results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_50_System_msg_3rd_prs/"
data["gpt35m"]["big5_resU2"] = "results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_50_User_msg_2nd_prs/"
data["gpt35m"]["big5_resU3"] = "results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_50_User_msg_3rd_prs/"

## GPT35j
data["gpt35j"]["pvq_resS2"] = "results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0613_perm_50_System_msg_2nd_prs/"
data["gpt35j"]["pvq_resS3"] = "results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0613_perm_50_System_msg_3rd_prs/"
data["gpt35j"]["pvq_resU2"] = "results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0613_perm_50_User_msg_2nd_prs/"
data["gpt35j"]["pvq_resU3"] = "results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0613_perm_50_User_msg_3rd_prs/"
data["gpt35j"]["hof_resS2"] = "results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0613_perm_50_System_msg_2nd_prs/"
data["gpt35j"]["hof_resS3"] = "results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0613_perm_50_System_msg_3rd_prs/"
data["gpt35j"]["hof_resU2"] = "results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0613_perm_50_User_msg_2nd_prs/"
data["gpt35j"]["hof_resU3"] = "results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0613_perm_50_User_msg_3rd_prs/"
data["gpt35j"]["big5_resS2"] = "results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0613_perm_50_System_msg_2nd_prs/"
data["gpt35j"]["big5_resS3"] = "results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0613_perm_50_System_msg_3rd_prs/"
data["gpt35j"]["big5_resU2"] = "results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0613_perm_50_User_msg_2nd_prs/"
data["gpt35j"]["big5_resU3"] = "results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0613_perm_50_User_msg_3rd_prs/"

## upstage llama 2
data["upllama2"]["pvq_resS2"] = "results_neurips/results_nat_lang_prof_pvq_test_up_llama2_70b_instruct_v2_perm_50_System_msg_2nd_prs/"
data["upllama2"]["pvq_resS3"] = "results_neurips/results_nat_lang_prof_pvq_test_up_llama2_70b_instruct_v2_perm_50_System_msg_3rd_prs/"
data["upllama2"]["pvq_resU2"] = "results_neurips/results_nat_lang_prof_pvq_test_up_llama2_70b_instruct_v2_perm_50_User_msg_2nd_prs/"
data["upllama2"]["pvq_resU3"] = "results_neurips/results_nat_lang_prof_pvq_test_up_llama2_70b_instruct_v2_perm_50_User_msg_3rd_prs/"
data["upllama2"]["hof_resS2"] = "results_neurips/results_nat_lang_prof_hofstede_test_up_llama2_70b_instruct_v2_perm_50_System_msg_2nd_prs/"
data["upllama2"]["hof_resS3"] = "results_neurips/results_nat_lang_prof_hofstede_test_up_llama2_70b_instruct_v2_perm_50_System_msg_3rd_prs/"
data["upllama2"]["hof_resU2"] = "results_neurips/results_nat_lang_prof_hofstede_test_up_llama2_70b_instruct_v2_perm_50_User_msg_2nd_prs/"
data["upllama2"]["hof_resU3"] = "results_neurips/results_nat_lang_prof_hofstede_test_up_llama2_70b_instruct_v2_perm_50_User_msg_3rd_prs/"
data["upllama2"]["big5_resS2"] = "results_neurips/results_nat_lang_prof_big5_test_up_llama2_70b_instruct_v2_perm_50_System_msg_2nd_prs/"
data["upllama2"]["big5_resS3"] = "results_neurips/results_nat_lang_prof_big5_test_up_llama2_70b_instruct_v2_perm_50_System_msg_3rd_prs/"
data["upllama2"]["big5_resU2"] = "results_neurips/results_nat_lang_prof_big5_test_up_llama2_70b_instruct_v2_perm_50_User_msg_2nd_prs/"
data["upllama2"]["big5_resU3"] = "results_neurips/results_nat_lang_prof_big5_test_up_llama2_70b_instruct_v2_perm_50_User_msg_3rd_prs/"

## upstage llama 1
data["upllama1"]["pvq_resS2"] = "results_neurips/results_nat_lang_prof_pvq_test_up_llama_60b_instruct_perm_50_System_msg_2nd_prs/"
data["upllama1"]["pvq_resS3"] = "results_neurips/results_nat_lang_prof_pvq_test_up_llama_60b_instruct_perm_50_System_msg_3rd_prs/"
data["upllama1"]["pvq_resU2"] = "results_neurips/results_nat_lang_prof_pvq_test_up_llama_60b_instruct_perm_50_User_msg_2nd_prs/"
data["upllama1"]["pvq_resU3"] = "results_neurips/results_nat_lang_prof_pvq_test_up_llama_60b_instruct_perm_50_User_msg_3rd_prs/"
data["upllama1"]["hof_resS2"] = "results_neurips/results_nat_lang_prof_hofstede_test_up_llama_60b_instruct_perm_50_System_msg_2nd_prs/"
data["upllama1"]["hof_resS3"] = "results_neurips/results_nat_lang_prof_hofstede_test_up_llama_60b_instruct_perm_50_System_msg_3rd_prs/"
data["upllama1"]["hof_resU2"] = "results_neurips/results_nat_lang_prof_hofstede_test_up_llama_60b_instruct_perm_50_User_msg_2nd_prs/"
data["upllama1"]["hof_resU3"] = "results_neurips/results_nat_lang_prof_hofstede_test_up_llama_60b_instruct_perm_50_User_msg_3rd_prs/"
data["upllama1"]["big5_resS2"] = "results_neurips/results_nat_lang_prof_big5_test_up_llama_60b_instruct_perm_50_System_msg_2nd_prs/"
data["upllama1"]["big5_resS3"] = "results_neurips/results_nat_lang_prof_big5_test_up_llama_60b_instruct_perm_50_System_msg_3rd_prs/"
data["upllama1"]["big5_resU2"] = "results_neurips/results_nat_lang_prof_big5_test_up_llama_60b_instruct_perm_50_User_msg_2nd_prs/"
data["upllama1"]["big5_resU3"] = "results_neurips/results_nat_lang_prof_big5_test_up_llama_60b_instruct_perm_50_User_msg_3rd_prs/"

## OpenAssistant
data["oa"]["pvq_resS2"]="results_neurips/results_nat_lang_prof_pvq_test_openassistant_rlhf2_llama30b_perm_50_System_msg_2nd_prs/"
data["oa"]["pvq_resS3"]="results_neurips/results_nat_lang_prof_pvq_test_openassistant_rlhf2_llama30b_perm_50_System_msg_3rd_prs/"
data["oa"]["pvq_resU2"]="results_neurips/results_nat_lang_prof_pvq_test_openassistant_rlhf2_llama30b_perm_50_User_msg_2nd_prs/"
data["oa"]["pvq_resU3"]="results_neurips/results_nat_lang_prof_pvq_test_openassistant_rlhf2_llama30b_perm_50_User_msg_3rd_prs/"
data["oa"]["hof_resS2"]="results_neurips/results_nat_lang_prof_hofstede_test_openassistant_rlhf2_llama30b_perm_50_System_msg_2nd_prs/"
data["oa"]["hof_resS3"]="results_neurips/results_nat_lang_prof_hofstede_test_openassistant_rlhf2_llama30b_perm_50_System_msg_3rd_prs/"
data["oa"]["hof_resU2"]="results_neurips/results_nat_lang_prof_hofstede_test_openassistant_rlhf2_llama30b_perm_50_User_msg_2nd_prs/"
data["oa"]["hof_resU3"]="results_neurips/results_nat_lang_prof_hofstede_test_openassistant_rlhf2_llama30b_perm_50_User_msg_3rd_prs/"
data["oa"]["big5_resS2"]="results_neurips/results_nat_lang_prof_big5_test_openassistant_rlhf2_llama30b_perm_50_System_msg_2nd_prs/"
data["oa"]["big5_resS3"]="results_neurips/results_nat_lang_prof_big5_test_openassistant_rlhf2_llama30b_perm_50_System_msg_3rd_prs/"
data["oa"]["big5_resU2"]="results_neurips/results_nat_lang_prof_big5_test_openassistant_rlhf2_llama30b_perm_50_User_msg_2nd_prs/"
data["oa"]["big5_resU3"]="results_neurips/results_nat_lang_prof_big5_test_openassistant_rlhf2_llama30b_perm_50_User_msg_3rd_prs/"

### StableVicuna
data["stvic"]["pvq_resU2"]="results_neurips/results_nat_lang_prof_pvq_test_stablevicuna_perm_50_User_msg_2nd_prs/"
data["stvic"]["pvq_resU3"]="results_neurips/results_nat_lang_prof_pvq_test_stablevicuna_perm_50_User_msg_3rd_prs/"
data["stvic"]["hof_resU2"]="results_neurips/results_nat_lang_prof_hofstede_test_stablevicuna_perm_50_User_msg_2nd_prs/"
data["stvic"]["hof_resU3"]="results_neurips/results_nat_lang_prof_hofstede_test_stablevicuna_perm_50_User_msg_3rd_prs/"
data["stvic"]["big5_resU2"]="results_neurips/results_nat_lang_prof_big5_test_stablevicuna_perm_50_User_msg_2nd_prs/"
data["stvic"]["big5_resU3"]="results_neurips/results_nat_lang_prof_big5_test_stablevicuna_perm_50_User_msg_3rd_prs/"

## StableLM
data["stlm"]["pvq_resS2"]="results_neurips/results_nat_lang_prof_pvq_test_stablelm_perm_50_System_msg_2nd_prs/"
data["stlm"]["pvq_resS3"]="results_neurips/results_nat_lang_prof_pvq_test_stablelm_perm_50_System_msg_3rd_prs/"
data["stlm"]["pvq_resU2"]="results_neurips/results_nat_lang_prof_pvq_test_stablelm_perm_50_User_msg_2nd_prs/"
data["stlm"]["pvq_resU3"]="results_neurips/results_nat_lang_prof_pvq_test_stablelm_perm_50_User_msg_3rd_prs/"
data["stlm"]["hof_resS2"]="results_neurips/results_nat_lang_prof_hofstede_test_stablelm_perm_50_System_msg_2nd_prs/"
data["stlm"]["hof_resS3"]="results_neurips/results_nat_lang_prof_hofstede_test_stablelm_perm_50_System_msg_3rd_prs/"
data["stlm"]["hof_resU2"]="results_neurips/results_nat_lang_prof_hofstede_test_stablelm_perm_50_User_msg_2nd_prs/"
data["stlm"]["hof_resU3"]="results_neurips/results_nat_lang_prof_hofstede_test_stablelm_perm_50_User_msg_3rd_prs/"
data["stlm"]["big5_resS2"]="results_neurips/results_nat_lang_prof_big5_test_stablelm_perm_50_System_msg_2nd_prs/"
data["stlm"]["big5_resS3"]="results_neurips/results_nat_lang_prof_big5_test_stablelm_perm_50_System_msg_3rd_prs/"
data["stlm"]["big5_resU2"]="results_neurips/results_nat_lang_prof_big5_test_stablelm_perm_50_User_msg_2nd_prs/"
data["stlm"]["big5_resU3"]="results_neurips/results_nat_lang_prof_big5_test_stablelm_perm_50_User_msg_3rd_prs/"

## LLaMa 65B
data["llama"]["pvq_resU2"]="results_neurips/results_nat_lang_prof_pvq_test_llama_65B_perm_50__msg_2nd_prs/"
data["llama"]["pvq_resU3"]="results_neurips/results_nat_lang_prof_pvq_test_llama_65B_perm_50__msg_3rd_prs/"
data["llama"]["hof_resU2"]="results_neurips/results_nat_lang_prof_hofstede_test_llama_65B_perm_50__msg_2nd_prs/"
data["llama"]["hof_resU3"]="results_neurips/results_nat_lang_prof_hofstede_test_llama_65B_perm_50__msg_3rd_prs/"
data["llama"]["big5_resU2"]="results_neurips/results_nat_lang_prof_big5_test_llama_65B_perm_50__msg_2nd_prs/"
data["llama"]["big5_resU3"]="results_neurips/results_nat_lang_prof_big5_test_llama_65B_perm_50__msg_3rd_prs/"

## RP Chat
data["rpchat"]["pvq_resU2"]="results_neurips/results_nat_lang_prof_pvq_test_rp_incite_7b_chat_perm_50_User_msg_2nd_prs/"
data["rpchat"]["pvq_resU3"]="results_neurips/results_nat_lang_prof_pvq_test_rp_incite_7b_chat_perm_50_User_msg_3rd_prs/"
data["rpchat"]["hof_resU2"]="results_neurips/results_nat_lang_prof_hofstede_test_rp_incite_7b_chat_perm_50_User_msg_2nd_prs/"
data["rpchat"]["hof_resU3"]="results_neurips/results_nat_lang_prof_hofstede_test_rp_incite_7b_chat_perm_50_User_msg_3rd_prs/"
data["rpchat"]["big5_resU2"]="results_neurips/results_nat_lang_prof_big5_test_rp_incite_7b_chat_perm_50_User_msg_2nd_prs/"
data["rpchat"]["big5_resU3"]="results_neurips/results_nat_lang_prof_big5_test_rp_incite_7b_chat_perm_50_User_msg_3rd_prs/"

## RP Instruct
data["rpinstruct"]["pvq_resU2"]="results_neurips/results_nat_lang_prof_pvq_test_rp_incite_7b_instruct_perm_50_User_msg_2nd_prs/"
data["rpinstruct"]["pvq_resU3"]="results_neurips/results_nat_lang_prof_pvq_test_rp_incite_7b_instruct_perm_50_User_msg_3rd_prs/"
data["rpinstruct"]["hof_resU2"]="results_neurips/results_nat_lang_prof_hofstede_test_rp_incite_7b_instruct_perm_50_User_msg_2nd_prs/"
data["rpinstruct"]["hof_resU3"]="results_neurips/results_nat_lang_prof_hofstede_test_rp_incite_7b_instruct_perm_50_User_msg_3rd_prs/"
data["rpinstruct"]["big5_resU2"]="results_neurips/results_nat_lang_prof_big5_test_rp_incite_7b_instruct_perm_50_User_msg_2nd_prs/"
data["rpinstruct"]["big5_resU3"]="results_neurips/results_nat_lang_prof_big5_test_rp_incite_7b_instruct_perm_50_User_msg_3rd_prs/"

## gpt-3.5-turbo-instruct
data["gpt35in"]["pvq_resU2"]="results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-instruct-0914_perm_50_User_msg_2nd_prs/"
data["gpt35in"]["pvq_resU3"]="results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-instruct-0914_perm_50_User_msg_3rd_prs/"
data["gpt35in"]["hof_resU2"]="results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-instruct-0914_perm_50_User_msg_2nd_prs/"
data["gpt35in"]["hof_resU3"]="results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-instruct-0914_perm_50_User_msg_3rd_prs/"
data["gpt35in"]["big5_resU2"]="results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-instruct-0914_perm_50_User_msg_2nd_prs/"
data["gpt35in"]["big5_resU3"]="results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-instruct-0914_perm_50_User_msg_3rd_prs/"

## Curie
data["curie"]["pvq_resU2"]="results_neurips/results_nat_lang_prof_pvq_test_curie_perm_50_User_msg_2nd_prs/"
data["curie"]["pvq_resU3"]="results_neurips/results_nat_lang_prof_pvq_test_curie_perm_50_User_msg_3rd_prs/"
data["curie"]["hof_resU2"]="results_neurips/results_nat_lang_prof_hofstede_test_curie_perm_50_User_msg_2nd_prs/"
data["curie"]["hof_resU3"]="results_neurips/results_nat_lang_prof_hofstede_test_curie_perm_50_User_msg_3rd_prs/"
data["curie"]["big5_resU2"]="results_neurips/results_nat_lang_prof_big5_test_curie_perm_50_User_msg_2nd_prs/"
data["curie"]["big5_resU3"]="results_neurips/results_nat_lang_prof_big5_test_curie_perm_50_User_msg_3rd_prs/"

## Babbage
data["babbage"]["pvq_resU2"]="results_neurips/results_nat_lang_prof_pvq_test_babbage_perm_50_User_msg_2nd_prs/"
data["babbage"]["pvq_resU3"]="results_neurips/results_nat_lang_prof_pvq_test_babbage_perm_50_User_msg_3rd_prs/"
data["babbage"]["hof_resU2"]="results_neurips/results_nat_lang_prof_hofstede_test_babbage_perm_50_User_msg_2nd_prs/"
data["babbage"]["hof_resU3"]="results_neurips/results_nat_lang_prof_hofstede_test_babbage_perm_50_User_msg_3rd_prs/"
data["babbage"]["big5_resU2"]="results_neurips/results_nat_lang_prof_big5_test_babbage_perm_50_User_msg_2nd_prs/"
data["babbage"]["big5_resU3"]="results_neurips/results_nat_lang_prof_big5_test_babbage_perm_50_User_msg_3rd_prs/"

## Ada
data["ada"]["pvq_resU2"]="results_neurips/results_nat_lang_prof_pvq_test_ada_perm_50_User_msg_2nd_prs/"
data["ada"]["pvq_resU3"]="results_neurips/results_nat_lang_prof_pvq_test_ada_perm_50_User_msg_3rd_prs/"
data["ada"]["hof_resU2"]="results_neurips/results_nat_lang_prof_hofstede_test_ada_perm_50_User_msg_2nd_prs/"
data["ada"]["hof_resU3"]="results_neurips/results_nat_lang_prof_hofstede_test_ada_perm_50_User_msg_3rd_prs/"
data["ada"]["big5_resU2"]="results_neurips/results_nat_lang_prof_big5_test_ada_perm_50_User_msg_2nd_prs/"
data["ada"]["big5_resU3"]="results_neurips/results_nat_lang_prof_big5_test_ada_perm_50_User_msg_3rd_prs/"


models = ["zephyr", "gpt4", "gpt35m", "gpt35j", "gpt35in", "upllama2","upllama1", "oa", "stvic", "stlm", "llama", "rpchat", "rpincite", "curie", "babbage", "ada"]
msg = ["S", "U"]
prs = ["2", "3"]

# pvq
questionnaires = ["pvq"]
comparisons = [("gpt35m", m) for m in models]
label_best = "pvq_resU2"

# hof
questionnaires = ["hof"]
comparisons = [("upllama1", m) for m in models]
label_best = "hof_resU3"
#
# # big5
questionnaires = ["big5"]
comparisons = [("gpt35j", m) for m in models]
label_best = "big5_resS3"


# replace paths with data from alignments.json
for model in models:
    for quest in questionnaires:
        for m in msg:

            for p in prs:
                label = quest + "_res" + m + p
                if label not in data[model]:
                    continue
                pa = Path(data[model][label])

                json_paths= list(pa.glob("*/alignments.json"))

                if len(json_paths) == 0:
                    print("No JSON files found in", pa)
                    continue

                json_data = []

                for json_path in json_paths:
                    # Open each JSON file
                    with open(json_path, 'r') as f:
                        # Load the JSON data from the file
                        load_data = json.load(f)
                        # Append the data to the list
                        json_data.extend(load_data)
                data[model][label] = json_data

p_limit = 0.05 / 15

print("p-limit: {}".format(p_limit))

for mod_1, mod_2 in comparisons:
    print("-> {} vs {}:".format(mod_1, mod_2))

    for quest in questionnaires:
        print(f"\t-> {quest}:")
        for m in msg:
            if (mod_1 == "stvic" or mod_2 == "stvic") and m == "S":
                continue

            for p in prs:
                label = quest + "_res" + m + p
                if label not in data[mod_1] or label not in data[mod_2]:
                    continue

                a=data[mod_1][label_best]
                b=data[mod_2][label]

                pvalue = stats.ttest_ind(a=a, b=b, equal_var=False).pvalue

                if pvalue < p_limit:
                    mark = "*"
                    color = "green"
                else:
                    mark = " "
                    color = "red"

                print(colored(f"\t {mark} {label.split('_')[1]} -> {pvalue}", color=color))