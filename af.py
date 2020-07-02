text = "aaa a,aa aabc."


def max_counts(text):
    counts={}
    for i in text.split(' '):
        if i in counts:
            counts[i] += 1
        else:
            counts[i] = 1
    return counts


text2 = max_counts(text)

print(text2)
    
