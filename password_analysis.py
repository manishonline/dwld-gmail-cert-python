import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from urllib.request import urlopen
import seaborn as sns


def run():
    url = 'http://www.dazzlepod.com/site_media/txt/passwords.txt'
    response = urlopen(url)
    print(response.info())

    file = open('passwords.txt', 'x')
    file.truncate()

    for line in response.read().decode('utf-8'):
        file.write(line)
    file.close()

    fh = open('passwords.txt', 'r+')

    for line in fh.readline()[:20]:
        print(line)

        # Test comment


if __name__ == '__main__':
    run()
