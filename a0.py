# 길이는 1 ~ 30
# 피타고라스의 정리를 만족
# 중복 제거
p = []
ap = []
i = 1
for c in range(30, 0, -1):
    for b in range(30, 0, -1):
        for a in range(30, 0, -1):
            if (a ** 2) == (b ** 2) + (c ** 2):
                p.append([a, b, c])
                p.sort()

print(p)
