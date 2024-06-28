import json
import random
def persona_description_to_name(simulated_participant_descr):
    return simulated_participant_descr.split(" (")[0]


# load persona descriptions
with open("real_world_people.txt", "r") as file:
    persona_descriptions = [l.rstrip() for l in file]

# load persona genders
with open("real_world_people_genders.txt", "r") as file:
    persona_genders = [l.rstrip() for l in file]

persona_names = [persona_description_to_name(desc) for desc in persona_descriptions]


personas = [{
    "name": n,
    "description": d,
    "gender": g
} for n, d, g in zip(persona_names, persona_descriptions, persona_genders)]

# Dumping personas to JSON Lines file
with open("personas.json", "w") as f:
    json.dump(personas, f, indent=4)


# create chunks with different permutations

personas_ = personas.copy()

import random


def permute_personas(personas):
    original_list = personas.copy()
    permuted_list = personas.copy()

    while True:
        random.shuffle(permuted_list)
        if not any(permuted_list[i]["name"] == original_list[i]["name"] for i in range(len(personas))):
            break

    return permuted_list


for chunk_i in range(5):
    perm_personas = permute_personas(personas)
    with open(f"./chunk_{chunk_i}/interlocutors.json", "w") as f:
        json.dump(perm_personas, f, indent=4)

