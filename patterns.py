import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from tqdm import tqdm

import os
if not os.path.exists('patterns'):
    os.mkdir('patterns')

def colorize_(i):
    if i==0:
        return to_rgba('k')
    else:
        return to_rgba(f'C{i-1}')

def colorize(arr):
    rgba = np.vectorize(colorize_)(arr)
    rgba = np.array(rgba).transpose(1,2,0)
    return rgba

def apgcode_to_array(s, Z='0'):
    while 'y' in s:
        i = s.find('y')
        s = s[:i] + 4*Z+Z*int(s[i+1],36) + s[i+2:]
    t = s.replace('w',2*Z).replace('x',3*Z).split('z')
    u = map(''.join,zip(*[[(Z*5+bin(int(c,32))[2:])[:-6:-1]for c in r]+[Z*5]*(max(map(len,t))-len(r))for r in t]))
    return list(u)

def plot_apgcode(apgcode):
    h, w = 0, 0
    arrs = []
    codes = apgcode.split('_')
    for i,code in enumerate(codes[1:]):
        arr = np.array([[c for c in row] for row in apgcode_to_array(code)], int).T
        h = max(h, arr.shape[0])
        w = max(w, arr.shape[1])
        arrs.append((i+1)*arr)
    arrs1 = []
    for arr in arrs:
        arr = np.pad(arr, [[1,h-arr.shape[0]+1],[1,w-arr.shape[1]+1]])
        arrs1.append(arr)
    sum_arr = np.sum(arrs1, 0)
    last_row = sum_arr.shape[0]
    for row in sum_arr[::-1]:
        if np.all(row==0):
            last_row -= 1
        else:
            break
    sum_arr = sum_arr[:last_row+1]
    fig = plt.figure()
    plt.axis('off')
    plt.imshow(colorize(sum_arr), aspect='equal')
    fig.tight_layout()
    return fig

cat_url = 'https://catagolue.hatsya.com'
hauls_url = cat_url+'/haul/xundead_b3s23/pcg64_16x16_1_1_1_stdin?committed=2'

hauls = pd.read_html(hauls_url)[0]
response = requests.get(hauls_url)
soup = BeautifulSoup(response.text, 'html.parser')
table = soup.find('table')
links = []
for tr in table.findAll('tr'):
    trs = tr.findAll('td')
    for each in trs:
        try:
            link = each.find('a')['href']
            links.append(link)
        except:
            pass
hauls['Link'] = links[::2]

patterns = np.array([])
for link in hauls['Link']:
    haul_url = cat_url+link
    haul = pd.read_html(haul_url)[0]
    pats = np.array([obj.split()[0] for obj in haul['Object']])
    pats = pats[np.logical_and(np.char.count(pats, '_')==2, np.array([obj.split('_')[1] for obj in pats])!='0')]
    patterns = np.union1d(patterns, pats)

for pattern in tqdm(patterns):
    try:
        fig = plot_apgcode(pattern)
        fig.savefig(f'patterns/{pattern}.png', bbox_inches='tight', pad_inches = 0)
        plt.close()
    except:
        print(pattern)
