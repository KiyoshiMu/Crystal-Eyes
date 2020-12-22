import requests

search_url = "http://127.0.0.1:8080/dectect/?topN=3"


def fetch():

    files = {"file": open("data/test0.jpg", "rb")}
    res = requests.post(search_url, files=files)
    print(res.text)


if __name__ == "__main__":
    fetch()
