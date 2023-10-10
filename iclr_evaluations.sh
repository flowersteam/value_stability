##GPTm

#PVQ
#python visualization_scripts/bar_viz.py results_iclr/results_pvq_test_sim_conv_gpt-3.5-turbo-0301_perm_50_theme/* --save --filename pvq_sim_conv_gpt35m_50
#python visualization_scripts/bar_viz.py results_iclr/results_pvq_test_format_gpt-3.5-turbo-0301_perm_50_format/* --save --filename pvq_formats_gpt35m_50
#python visualization_scripts/bar_viz.py results_iclr/results_AI_wiki_context_v2_no_separator_music_expert_pvq_test_gpt-3.5-turbo-0301_perm_50_User_msg_3rd_prs/* --save --filename pvq_wiki_gpt35m_50

## HOF
#python visualization_scripts/bar_viz.py results_iclr/results_hofstede_test_sim_conv_gpt-3.5-turbo-0301_perm_50_theme/* --save --filename hof_sim_conv_gpt35m_50
#python visualization_scripts/bar_viz.py results_iclr/results_hofstede_test_format_gpt-3.5-turbo-0301_perm_50_format/* --save --filename hof_formats_gpt35m_50
#python visualization_scripts/bar_viz.py results_iclr/results_AI_wiki_context_v2_no_separator_music_expert_hofstede_test_gpt-3.5-turbo-0301_perm_50_User_msg_3rd_prs/* --save --filename hof_wiki_gpt35m_50

# BIG5
python visualization_scripts/bar_viz.py results_iclr/results_big5_test_sim_conv_gpt-3.5-turbo-0301_perm__theme/* --save --filename big5_sim_conv_gpt35m_50
python visualization_scripts/bar_viz.py results_iclr/results_big5_test_format_gpt-3.5-turbo-0301_perm_50_format/* --save --filename big5_formats_gpt35m_50
python visualization_scripts/bar_viz.py results_iclr/results_AI_wiki_context_v2_no_separator_music_expert_big5_test_gpt-3.5-turbo-0301_perm_50_User_msg_3rd_prs/* --save --filename big5_wiki_gpt35m_50

exit


#GPTj
# PVQ
#python visualization_scripts/bar_viz.py results_iclr/results_pvq_test_sim_conv_gpt-3.5-turbo-0613_perm_50_theme/* --save --filename pvq_sim_conv_gpt35j_50
#python visualization_scripts/bar_viz.py results_iclr/results_pvq_test_format_gpt-3.5-turbo-0613_perm_50_format/* --save --filename pvq_formats_gpt35j_50
#python visualization_scripts/bar_viz.py results_iclr/results_AI_wiki_context_v2_no_separator_music_expert_pvq_test_gpt-3.5-turbo-0613_perm_50_User_msg_3rd_prs/* --save --filename pvq_wiki_gpt35j_50

# HOF
#python visualization_scripts/bar_viz.py results_iclr/results_hofstede_test_sim_conv_gpt-3.5-turbo-0613_perm_50_theme/* --save --filename hof_sim_conv_gpt35j_50
#python visualization_scripts/bar_viz.py results_iclr/results_hofstede_test_format_gpt-3.5-turbo-0613_perm_50_format/* --save --filename hof_formats_gpt35j_50
#python visualization_scripts/bar_viz.py results_iclr/results_AI_wiki_context_v2_no_separator_music_expert_hofstede_test_gpt-3.5-turbo-0613_perm_50_User_msg_3rd_prs/* --save --filename hof_wiki_gpt35j_50



# NOTE: remember to change ylims in bar_viz.py
#        if test_set_name == "pvq_male":
#            ax.set_ylim([-3, 3]) # append
#            # ax.set_ylim([-2.5, 2.5])
#
#        elif test_set_name == "hofstede":
#            ax.set_ylim([-350, 350]) # append
#            # ax.set_ylim([-150, 150])
#
#        elif test_set_name == "big5_50":
#            ax.set_ylim([0, 55])
#
#        elif test_set_name == "big5_100":
#            ax.set_ylim([0, 110])

# gpt4
pvq_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-4-0314_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/pvq_gpt4_50_S3`
hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-4-0314_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt4_50_U3`
big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-4-0314_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt4_50_U3`

# gpt35m
pvq_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0301_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35m_50_S2`
hof_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0301_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/hof_gpt35m_50_U2`
big5_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0301_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/big5_gpt35m_50_U2`

# gpt35j
pvq_resS2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_gpt-3.5-turbo-0613_perm_50_System_msg_2nd_prs/* --save --filename neurips_plots/pvq_gpt35j_50_S2`
hof_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_gpt-3.5-turbo-0613_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/hof_gpt35j_50_S3`
big5_resS3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_gpt-3.5-turbo-0613_perm_50_System_msg_3rd_prs/* --save --filename neurips_plots/big5_gpt35j_50_S3`

# Upllama
pvq_resU2=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_pvq_test_up_llama_60b_instruct_perm_50_User_msg_2nd_prs/* --save --filename neurips_plots/pvq_upllama_50_U2`
hof_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_hofstede_test_up_llama_60b_instruct_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/hof_upllama_50_U3`
big5_resU3=`python visualization_scripts/bar_viz.py results_neurips/results_nat_lang_prof_big5_test_up_llama_60b_instruct_perm_50_User_msg_3rd_prs/* --save --filename neurips_plots/big5_upllama_50_U3`
