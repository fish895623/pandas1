

def max_counts(text):
    counts={}
    for i in text.split(' '):
        if i in counts:
            counts[i] += 1
        else:
            counts[i] = 1
    return counts

        
People={"홍", "홍","김"}

aabv = {"abcd":2, "aaaa":3}

print(max(aabv.values()))
