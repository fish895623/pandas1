import random


def main():
    # 남은 횟수 출력
    # 사용단어 리스트 출력
    # 맞춘 단어 까기

    # 스트링 전체를 바꾸기
    # if "입력한것" in 스트링
    count = 10
    word_list = ["abcd", "sbcdfge"]
    word = word_list[int(random.uniform(0, len(word_list)))].lower()

    a = input("asdfadf")
    for i in range(len(word)):
        j = word[i]
        if a == j:
            print("a")
        elif a != j:
            count -= 1


if __name__ == "__main__":
    main()
