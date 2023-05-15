# PVQ (lotr) - one eval:
# total GPT tokens used: 5200
#         gpt-4 ~ 0.2080 dollars
#         gpt-3.5 ~ 0.0104 dollars
#         davinci ~ 0.1040 dollars
#         curie ~ 0.0104 dollars
#         babagge ~ 0.0026 dollars
#         ada ~ 0.0021 dollars

# PVQ (high detail) - one eval:
# total GPT tokens used: 7840
#         gpt-4 ~ 0.3136 dollars
#         gpt-3.5 ~ 0.0157 dollars
#         davinci ~ 0.1568 dollars
#         curie ~ 0.0157 dollars
#         babagge ~ 0.0039 dollars
#         ada ~ 0.0031 dollars


# one eval

# n_tokens_per_persp = 5200
# gpt_4 = 0.208
# gpt_35 = 0.01
# davinci = 0.1

# n_tokens_per_persp = 7840
# gpt_4 = 0.3136
# gpt_35 = 0.0157
# davinci = 0.1568

# n_tokens_per_persp = 5560
# gpt_4 = 0.2224
# gpt_35 = 0.0111
# davinci = 0.1112

# price per token
gpt_4 = 0.03/1000
gpt_35 = 0.002/1000
davinci = 0.02/1000


# 1. PVQ: lotr 5 + prim 0  -> 5 ( prim is in 3.)
# exp1 = 0
exp1 = 5 * 4640

# 2. PVQ:music 6 +  hobbies: 5 -> 11
# exp2 = 0
exp2 = 6 * 4600 + 5 * 4520

# 3. message person: (PVQ: 4 HOF: 6 B5: 5) x 4 settings = 15*4 -> 60
pvq_3 = 4*4*5040*0
hof_3 = 4*6*2200*0
big5_3 = 4*5*3500*0 # 50 items
big5_100_3 = 4*5*7083*0  # 100 items

exp3 = pvq_3 + hof_3 + big5_3 + big5_100_3

# 4. smooth: (PVQ: 4 HOF: 6 B5: 5) x 2 settings = 15*2 -> 30 (one is covered in 3.)
# pvq = 2* 4 * 5320  # 2nd system
pvq_4 = 2*4*5760*0
hof_4 = 2*6*2200*0
big5_4 = 2*5*3500*0  # 50 items
big5_100_4 = 2*5*7083*0  # 100 items
exp4 = pvq_4 + hof_4 + big5_4 + big5_100_4


n_permutations = 5
print("n_permutations:", n_permutations)

total_persp_x_tokens = sum([exp1, exp2, exp3, exp4])

total_persp_x_tokens = exp3

total_tokens = n_permutations * total_persp_x_tokens

price_gpt4 = gpt_4 * total_tokens
price_gpt35 = gpt_35 * total_tokens
price_davinci = davinci * total_tokens
print(f"Total price:\n\tGPT4: {price_gpt4}\n\tGPT35: {price_gpt35}\n\tDavinci: {price_davinci}")