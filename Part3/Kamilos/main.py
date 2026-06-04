from rank import rank


if __name__ == "__main__":
    query = "Najtańsze możliwe gpu nvidia z 5.3gb vram do prostych testów."

    print(rank(query))


    # while True:
    #     print("Podaj zdanie dla naszego najnowszego cutting-edge AI:")
    #     user_input = input()
    #
    #     result_struct = model.extract_gpu_criteria(user_input)
    #
    #     print("Wyciągnięta struktura danych:")
    #     print(json.dumps(result_struct, indent=2, ensure_ascii=False))
