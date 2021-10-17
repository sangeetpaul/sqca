import os
import numpy as np
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from tqdm import tqdm


def scrape_links(url):
    response = requests.get(url)
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
    return np.array(links)


def colorize_(i):  # convert 0,1,2 to k,C0,C1
    if i == 0:
        return to_rgba('k')
    else:
        return to_rgba(f'C{i - 1}')


def colorize(arr):  # convert 2d array to RGBA array
    rgba = np.vectorize(colorize_)(arr)
    rgba = np.array(rgba).transpose((1, 2, 0))
    return rgba


def apgcode_to_array(s):
    while 'y' in s:
        i = s.find('y')
        s = s[:i] + 4 * '0' + '0' * int(s[i + 1], 36) + s[i + 2:]
    t = s.replace('w', 2 * '0').replace('x', 3 * '0').split('z')
    u = map(''.join,
            zip(*[[('0' * 5 + bin(int(c, 32))[2:])[:-6:-1] for c in r] + ['0' * 5] * (max(map(len, t)) - len(r)) for r in
                  t]))
    return list(u)


def plot_apgcode(apgcode):
    h, w = 0, 0
    arrays = []
    codes = apgcode.split('_')
    for i, code in enumerate(codes[1:]):
        arr = np.array([[c for c in row] for row in apgcode_to_array(code)], int).T
        h = max(h, arr.shape[0])
        w = max(w, arr.shape[1])
        arrays.append((i + 1) * arr)
    arrays1 = []
    for arr in arrays:
        arr = np.pad(arr, [[1, h - arr.shape[0] + 1], [1, w - arr.shape[1] + 1]])
        arrays1.append(arr)
    sum_arr = np.sum(arrays1, 0)
    last_row = sum_arr.shape[0]
    for row in sum_arr[::-1]:
        if np.all(row == 0):
            last_row -= 1
        else:
            break
    sum_arr = sum_arr[:last_row + 1]
    fig = plt.figure()
    plt.axis('off')
    plt.imshow(colorize(sum_arr), aspect='equal')
    fig.tight_layout()
    return fig


cat_url = 'https://catagolue.hatsya.com'

# scrape Catagolue for hauls
hauls_url = cat_url + '/haul/xundead_b3s23/pcg64_16x16_1_1_1_stdin?committed=2'
haul_urls = scrape_links(hauls_url)[::2]

# scrape hauls for "interesting" patterns
patterns = np.array([])
for url in haul_urls:
    haul_url = cat_url + url
    links = scrape_links(haul_url)
    pat_urls = links[['object' in link for link in links]]
    pats = np.array([obj.split('/')[2] for obj in pat_urls])
    pats = pats[np.logical_and(np.char.count(pats, '_') == 2, np.array([obj.split('_')[1] for obj in pats]) != '0')]
    patterns = np.union1d(patterns, pats)

# save patterns as images
if not os.path.exists('patterns'):
    os.mkdir('patterns')
for pattern in tqdm(patterns):
    try:
        category = pattern.split('_')[0]
        if os.path.exists(f'patterns/{category}/{pattern}.png'):
            continue
        if not os.path.exists(f'patterns/{category}'):
            os.mkdir(f'patterns/{category}')
        fig = plot_apgcode(pattern)
        fig.savefig(f'patterns/{category}/{pattern}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
    except:
        print(f'Could not save {pattern}')
