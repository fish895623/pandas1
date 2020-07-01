words = ["aaaabcd", "sbcdfge"]
b = 0
for j in range(len(words[0])):
    try:
        print(words[0].index('a', b, -1))
    except ValueError:
        break
    b += 1
