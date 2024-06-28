
def simulated_participant_to_name(simulated_participant, simulated_population_config):

    if simulated_population_config in ["tolkien_characters"]:
        return simulated_participant

    elif simulated_population_config in ["famous_people", "real_world_people"]:
        # e.g. "Marilyn Monroe (1926 – 1962) American actress, singer, model."
        return simulated_participant.split(" (")[0]

    elif simulated_population_config == "permutations":
        return "CHATBOT"

    else:
        raise NotImplementedError(f"Participant name not implemented for population {simulated_population_config}.")
