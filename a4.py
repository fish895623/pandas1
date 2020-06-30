# 4 배수 제외
# 100 제외
# 400 나누어 떨어짐 제외
year = int(input())

if ((year % 4) == 0) and ((year % 100) != 0) and ((year % 400) != 0):
    print("a")
else:
    print("b")
