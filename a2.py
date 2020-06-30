f = open("ly.txt", "r")
lines = f.readlines()
table = dict()
for line in lines:
    line = (
        line.replace(",", "")
        .replace("don't", "do not")
        .replace("i'm", "i am")
        .replace("there's", "there is")
        .replace("wouldn't", "would not")
        .lower()
    )
    words = line.split()
    for word in words:
        try:
            table[word] += 1
        except:
            table[word] = 1

print(table)
f.close()
