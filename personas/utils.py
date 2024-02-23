
def simulated_participant_to_name(simulated_participant, simulated_population_type):

    if simulated_population_type in ["tolkien_characters"]:
        return simulated_participant

    elif simulated_population_type == "famous_people":
        # e.g. "Marilyn Monroe (1926 – 1962) American actress, singer, model."
        return simulated_participant.split(" (")[0]

    elif simulated_population_type == "permutations":
        return "CHATBOT"

    elif simulated_population_type == "anes":
        raise NotImplementedError(f"Participant name not implemented for population {simulated_population_type}.")

    elif simulated_population_type == "llm_personas":
        raise NotImplementedError(f"Participant name not implemented for population {simulated_population_type}.")

    elif simulated_population_type == "user_personas":
        raise NotImplementedError(f"Participant name not implemented for population {simulated_population_type}.")

    else:
        raise NotImplementedError(f"Participant name not implemented for population {simulated_population_type}.")



