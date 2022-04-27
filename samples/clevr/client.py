import requests

url = 'http://localhost:8080'
myobj = {'somekey': 'somevalue'}

x = requests.post(url, data = myobj)

print(x.text)