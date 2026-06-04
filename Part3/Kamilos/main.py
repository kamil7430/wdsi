from rank import rank


if __name__ == "__main__":
    # query = "Najtańsze możliwe gpu nvidia z 5.3gb vram do prostych testów."
    #
    # print(rank(query))

    while True:
        print("Podaj zdanie dla naszego najnowszego cutting-edge AI: ", end="")
        query = input()
        print("Wyciągnięta struktura danych:")
        print(rank(query))

