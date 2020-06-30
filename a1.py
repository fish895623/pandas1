d = 90

for a in range(1, d):
    for b in range(1, d):
        c = d - a - b

        if a ** 2 + b ** 2 == c ** 2:
            print(a, b, c)
