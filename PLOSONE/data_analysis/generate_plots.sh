##!/bin/bash
#
## rank-order
#figure_names=(
#  tolk_ro_t
#  fam_ro_t
#  don_t
#  bag_t
#  religion_t
#  paired_tolk_ro_uni
#  paired_tolk_ro_ben
#  paired_tolk_ro_pow
#  paired_tolk_ro_ach
#  no_pop_ips
#  tolk_ro_msgs_more
#)
#
#for fig_name in "${figure_names[@]}"; do
#  python PLOSONE/data_analysis/campaign_data_analysis.py --fig-name $fig_name --no-show
#done

# ipsative
python PLOSONE/data_analysis/campaign_data_analysis_ips_msgs.py --fig-name tolk_ips_msgs  --no-show # creates "tolkien_ipsative_curve_cache.json"
python PLOSONE/data_analysis/campaign_data_analysis_ips_msgs.py --fig-name ips_msgs --no-show
