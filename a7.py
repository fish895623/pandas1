#!/bin/python3


def main():
    num = int(input("정수를 입력하시오: "))
    print(is_prime(num))


def is_prime2(i):
    for j in range(2, i):
        if i % j == 0:
            return False
        return True


def is_prime(i):
    number_list = list(range(2, i))
    for j in range(len(number_list)):
        if i % number_list[j] == 0:
            number_list[j] = True
        else:
            number_list[j] = False

    if True in number_list:
        return True
    else:
        return False


if __name__ == "__main__":
    main()
