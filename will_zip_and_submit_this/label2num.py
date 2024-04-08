def label2num(label):
    classes = {
        "blues": 1,
        "classical": 2,
        "country": 3,
        "disco": 4,
        "hiphop": 5,
        "jazz": 6,
        "metal": 7,
        "pop": 8,
        "reggae": 9,
        "rock": 10
    }
    return classes.get(label, -1)  # return -1 if label is not found
