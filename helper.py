import json
import os


def get_settings():
    with open('./settings.json') as json_file:
        return json.load(json_file)
        

# def main():
#     get_settings()


# if __name__ == '__main__':
#     main()