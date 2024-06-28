from itertools import chain
import random
import csv
# https://tolkiengateway.net/wiki/Portal:Characters

# ELVES
elves_female = ["Nielthi", "Melian", "Míriel", "Tatië", "Evranin", "Nellas", "Aredhel", "Anairë", "Meleth", "Indis"]
elves_male = ["Círdan","Arminas","Elladan", "Maedhros", "Pengolodh", "Olwë", "Eöl", "Beleg", "Thingol", "Narthseg"]
selected_elves = elves_female+elves_male
assert len(selected_elves) == 20

# DWARVES
dwarves_female = ["Dís"]
dwarves_male =['Anar', 'Azaghâl', 'Bombur', 'Dáin', 'Fangluin', 'Flói', 'Frerin', 'Frár', 'Fundin', 'Gamil', 'Glóin', 'Ibun', 'Khîm', 'Kíli', 'Naugladur', 'Nori', 'Náin', 'Thrór', 'Óin']
selected_dwarves = dwarves_female+dwarves_male
assert len(selected_dwarves) == 20

# ORCS
orcs_female = []
orcs_male = ['Azog', 'Balcmeg',  'Bolg', 'Golfimbul', 'Gorgol', 'Grishnákh', 'Hobgoblins', 'Lagduf', 'Lug', 'Lugdush', 'Mauhúr', 'Muzgash', 'Naglur-Danlo', 'Orcobal', 'Othrod', 'Radbug', 'Shagram', 'Shagrat', 'Ufthak', 'Uglúk']
selected_orcs = orcs_female+orcs_male
assert len(selected_orcs) == 20

# HUMANS
humans_female = ["Lothíriel", "Almáriel", "Hiril", "Rían", "Ioreth", "Morwen", "Vidumavi", "Fíriel", "Zamîn", "Ivorwen"]
humans_male = ["Ohtar", "Beregar", "Ulbar", "Haldad", "Radhruin", "Fíriel", "Marhwini", "Valandur", "Anborn", "Estelmo"]
selected_humans = humans_female+humans_male
assert len(selected_humans) == 20

# HOBBITS
hobbits_female = ["Druda", "Donnamira", "Jessamine", "Melilot", "Ivy", "Gilly", "Belba", "Cora", "Lalia", "Mirabella"]
hobbits_male = ["Longo", "Adalgar", "Minto", "Bosco", "Drogo", "Ferdinand", "Gaffer", "Fortinbras", "Hildifons", "Iago"]
selected_hobbits = hobbits_female+hobbits_male
assert len(selected_hobbits) == 20


# remove .new manually
csv_file = open("data/data_tolkien_donation/test/tolkien_donation_test.csv.new", "w")
writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_ALL)


def add_genders(names, gender, race):
    return list(zip(names, [gender]*len(names), [race]*len(names)))

names_genders = list(chain(
    add_genders(elves_male[:5], "M", "elf"),
    add_genders(elves_female[:5], "F", "elf"),
    add_genders(elves_male[5:], "M", "elf"),
    add_genders(elves_female[5:], "F", "elf"),
    add_genders(dwarves_male, "M", "dwarf"),  # only one female so no alternating mfmf is needed
    add_genders(dwarves_female, "F", "dwarf"),
    add_genders(orcs_male, "M", "orc"),
    add_genders(orcs_female, "F", "orc"), # no female orcs so no alternating mfmf is needed
    add_genders(humans_male[:5], "M", "human"),
    add_genders(humans_female[:5], "F", "human"),
    add_genders(humans_male[5:], "M", "human"),
    add_genders(humans_female[5:], "F", "human"),
    add_genders(hobbits_male[:5], "M", "hobbit"),
    add_genders(hobbits_female[:5], "F", "hobbit"),
    add_genders(hobbits_male[5:], "M", "hobbit"),
    add_genders(hobbits_female[5:], "F", "hobbit"),
))
assert len(names_genders) == 100

for name_i, (name, gender, race) in enumerate(names_genders):

    if gender == "M":
        # male
        pronoun = "he"
        pos_pronoun = "his"
        obj_pronoun = "him"
        refl_pronoun = "himself"
    else:
        # female
        pronoun = "she"
        pos_pronoun = "her"
        obj_pronoun = "her"
        refl_pronoun = "herself"

    take_options = [0, 1, 2, 3, 4] * 2 + [6, 7, 8, 9, 10] * 2
    assert len(take_options) == 20
    take = take_options[name_i % 20]

    offer = 10 - take

    story = f"There is a poor hungry {race} called {name} next to you. {pronoun.capitalize()} asks you for some coins so that {pronoun} can feed {refl_pronoun} and {pos_pronoun} family. You have 10 silver coins in your pouch. How much do you give {obj_pronoun}?"

    writer.writerow([story, "0 coins", "2 coins", "4 coins", "6 coins", "8 coins", "10 coins", "undef"])
    print(story)
