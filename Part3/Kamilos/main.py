import json

import model

if __name__ == "__main__":
    while True:
        print("Podaj zdanie dla naszego najnowszego cutting-edge AI:")
        user_input = input()

        result_struct = model.extract_gpu_criteria(user_input)

        print("Wyciągnięta struktura danych:")
        print(json.dumps(result_struct, indent=2, ensure_ascii=False))