dic = {"name": [1, 5, 10]}

#
# class FourCal:
#     name = ""
#     first = 0
#     second = 0
#     third = 0

#     def div(self, dictionary, name):
#         dic[name] = [1, 2, 3]
#         return dic


# if __name__ == "__main__":
#     dic = {"name": [1, 5, 10]}
#     aaaa = FourCal()
    # print(aaaa.div(dic, "sss"))


class classname:
    def __init__(self, dic, name, kor, eng, _math):
        self.kor = 0
        self.eng = 0
        self._math = 0

    def funcname(parameter_list):
        pass


def main():
    try:
        while True:
            select = int(input("1입력 2출력 3검색 4삭제 5수정 6종료\n"))
            if select == 1:
                inputScore()
            elif select == 2:
                printScore()
            elif select == 3:
                print("abc")
                # elif select == 4:
                # elif select == 5:
            elif select == 6:
                break
            else:
                print("you put wrong num...")
            quit()
    except SystemExit:
        print("exit")


def inputScore():
    global dic
    name = str(input("이름을 적으세요"))
    dic[name], dic["국"], dic["영"], dic["수"], dic["총합"], dic["평균"]


def printScore():
    global dic
    print("이름\t: [ 국\t 영\t 수\t 총합 평균]")
    print(
        "%s\t: [ %d\t %d\t %d\t %d\t %d ]"
        % (dic["name"], dic["국"], dic["영"], dic["수"], dic["총합"], dic["평균"])
    )


if __name__ == "__main__":
    main()

# 입 출력 검색 삭제 수정 종료
