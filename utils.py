import os

def print_chat_messages(messages):
    print("*********************")
    print("Messages:")
    for msg in messages:
        print(f"{msg['role'].upper()} : {msg['content']}")
    print("*********************")


def remove_prefix(s, pref):
    if s.startswith(pref):
        return s[len(pref):]
    return s

def extract_answer_tokens(answers, tokenizer):
    answer_tokens = {a: [] for a in answers}
    for tok_ind in range(len(tokenizer)):
        tok = tokenizer.decode([tok_ind])
        if tok in answers:
            answer_tokens[tok].append(tok_ind)

    return answer_tokens

def get_hf_cache_dir():
    hostname = os.uname()[1]
    if hostname == "PTB-09003439":
        hf_cache_dir = "/home/flowers-user/.cache/huggingface"
    elif "plafrim" in hostname:
        hf_cache_dir ="/beegfs/gkovac/hf_cache_dir"
    else:
        hf_cache_dir = "/gpfsscratch/rech/imi/utu57ed/.cache/huggingface"
    return hf_cache_dir
def estimate_and_print_gpt_prices(gpt_tokens_count, engine):
    assert gpt_tokens_count.keys() == {"input", "output"}

    gpt_prices = {
        # model, inp_1M, out_1M
        "gpt-4-turbo-2024-04-09": (10, 30),  # best, biggest context size and the cheapest
        "gpt-4": (30, 60),
        "gpt-4-32k": (60, 120),
        "gpt-4-0125-preview": (10.00, 30.00),
        "gpt-4-1106-preview": (10.00, 30.00),
        "gpt-4-vision-preview": (10.00, 30.00),
        "gpt-3.5-turbo-0125": (0.5, 1.5),  # flagship
        "gpt-3.5-turbo-1106": (1.00, 2.00),
        "gpt-3.5-turbo-0613": (1.50, 2.00),
        "gpt-3.5-turbo-16k-0613": (3.00, 4.00),
        "gpt-3.5-turbo-0301": (1.50, 2.00),
        "gpt-3.5-turbo-instruct": (1.5, 2.0),
    }

    inp_toks, out_toks = gpt_tokens_count['input'], gpt_tokens_count['output']
    tot_toks = inp_toks + out_toks
    print(f"total GPT tokens used: {tot_toks} (in: {inp_toks} out: {out_toks})")

    engines_to_show = [
        "gpt-4-turbo-2024-04-09",
        "gpt-4-0125-preview",
        "gpt-4-1106-preview",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-0301",
    ]
    if engine in gpt_prices.keys() and engine not in engines_to_show:
        engines_to_show.insert(0, engine)

    for gpt_eng in engines_to_show:
        price_1M_inp, price_1M_out = gpt_prices[gpt_eng]
        price_input = (inp_toks / 1_000_000) * price_1M_inp
        price_ouput = (out_toks / 1_000_000) * price_1M_out
        tot_price = price_input + price_ouput
        print(f"\t{gpt_eng} ~ {tot_price:.2f}$ (in: {price_input:.2f}$ out: {price_ouput:.2f}$)")

openai_2_azure_tag = {
    "gpt-3.5-turbo-0125": "gpt-35-turbo-0125",
    "gpt-3.5-turbo-1106": "gpt-35-turbo-1106"
}