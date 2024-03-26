def num2label(number):
    classes = {
        1: "blues",
        2: "classical",
        3: "country",
        4: "disco",
        5: "hiphop",
        6: "jazz",
        7: "metal",
        8: "pop",
        9: "reggae",
        10: "rock"
    }
    return classes.get(number, "Unknown") # return "Unknown" if number not found

