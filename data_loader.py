import bisect
import math

import numpy as np
import scipy.sparse as sp

from PIL import Image
import csv, json
from tqdm import tqdm

# generate random matrix
def load_random(m=2048, n=1024, fill_ratio=0.1, rd_seed=None) -> np.ndarray:
	# for reproducibility
	if rd_seed is not None:
		np.random.seed(rd_seed)

	A = np.random.rand(m, n)
	A = np.array(A<fill_ratio, dtype=np.int32)
	return A

# load an image as matrix
def load_image(image_path, binary=True) -> np.ndarray:
	img = Image.open(image_path)
	img = img.convert('L')
	img_data = np.array(img)
	A = np.array(img_data/255, dtype=np.float64)
	if binary:
		A = np.round(A)
	#plt.imshow(A, cmap='gray')
	return A

def idx(val, arr): # find val in arr, based on binary search
	res = bisect.bisect_left(arr, val)
	if res == len(arr) or arr[res] != val: return None
	return res

#* Different versions of MovieLens have different formats;
#* So we need to prepare different loader functions

# load MovieLen-latest-small or MovieLen-25m
def load_movielen(dir='.', sample_ratio=1, rd_seed=None, binary=True) -> sp.csr_matrix:
	with open(dir+'/manifest.json', 'r', encoding='utf-8') as f:
		metadata = json.load(f)
	nr_movies = metadata['nr_movies']
	nr_users  = metadata['nr_users']
	nr_records= metadata['nr_records']

	# for reproducibility
	if rd_seed is not None:
		np.random.seed(rd_seed)

	# load movies and their id(mid=movie id)
	with open(dir+'/movies.csv', 'r', encoding='utf-8') as f:
		reader = csv.reader(f)
		reader.__next__()
		all_mid = np.array([int(row[0]) for row in reader])

	# random selected users and movies, in userId and movieId
	selected_users = sorted(np.random.choice(nr_users,  math.floor(nr_users *sample_ratio), replace=False)+1)# uid starts from 1
	selected_movies= sorted(np.random.choice(nr_movies, math.floor(nr_movies*sample_ratio), replace=False)  )# np.floor is still float
	selected_movies= all_mid[selected_movies] # mid are not continuous

	# dimension of the downsampled preference matrix
	m, n = len(selected_users), len(selected_movies)
	"""A = np.zeros((m, n), dtype=np.float32) # single precision is enough

	# load main table
	low_mask = (1<<16)-1
	with open(dir+'/ratings.csv', 'r', encoding='utf-8') as f:
		reader = csv.reader(f)
		reader.__next__() # consume title row
		rec_cnt = 1
		for row in reader:
			uid = int(row[0]) # user id
			mid = int(row[1]) # movie id
			rating = float(row[2])

			i = idx(uid, selected_users)
			j = idx(mid, selected_movies)
			if (i is not None) and (j is not None): # can't abbreviate; i and j can be 0
				A[i,j] = rating

			rec_cnt += 1
			if rec_cnt & low_mask == 0: # per 16384 records - bit ops should be faster than mod
				print(f'\rLoading movielen dataset {rec_cnt/nr_records*100:.2f}%...', end='')
		print('\rLoading done!', end='')

	if binary:
		return np.array(A>=3, dtype=np.int32)
	else:
		return A
		"""

	low_mask = (1<<16)-1
	rows = []; cols = []; data = []
	with open(dir+'/ratings.csv', 'r', encoding='utf-8') as f:
		reader = csv.reader(f)
		reader.__next__() # consume title row
		rec_cnt = 0
		for row in reader:
			rec_cnt += 1
			if rec_cnt & low_mask == 0: # per 16384 records - bit ops should be faster than mod
				print(f'\rLoading movielen dataset {rec_cnt/nr_records*100:.2f}%...', end='')

			uid = int(row[0]) # user id
			mid = int(row[1]) # movie id
			rating = float(row[2])

			i = idx(uid, selected_users)
			j = idx(mid, selected_movies)
			# can't abbreviate; i and j can be 0
			if (i is None) or (j is None):
				continue

			rows.append(i)
			cols.append(j)
			data.append(rating if not binary else (rating>=3))

	A = sp.csr_matrix((data, (rows, cols)), shape=(m, n), dtype=np.float32)
	print('\rLoading done!', end='')

	return A

load_movielen_latest_small = load_movielen

def load_movielen_100k(dir='.', sample_ratio=1, rd_seed=None, binary=True):
	# for reproducibility
	if rd_seed is not None:
		np.random.seed(rd_seed)

	with open(dir+'/u.data', 'r', encoding='utf-8') as f:
		records = f.readlines()
	m, n = 943, 1682 # #user and #movie can be found in readme file
	A = np.zeros((m, n), dtype=np.float32)
	for rec in records:
		if sample_ratio < 1 and np.random.rand() > sample_ratio: continue
		rec = rec.strip()
		uid, mid, rating, timestamp = rec.split('\t')
		uid = int(uid) - 1
		mid = int(mid) - 1
		rating = int(rating)

		A[uid, mid] = rating

	if binary: A = (A>=3).astype(np.float32)
	return A

def load_movielen_1m(dir='.', sample_ratio=1, rd_seed=None, binary=True):
	# for reproducibility
	if rd_seed is not None:
		np.random.seed(rd_seed)

	with open(dir+'/ratings.dat', 'r', encoding='utf-8') as f:
		records = f.readlines()
	m, n = 6040, 3952 # #user and #movie can be found in readme file
	A = np.zeros((m, n), dtype=np.float32)
	for rec in records:
		if sample_ratio < 1 and np.random.rand() > sample_ratio: continue
		rec = rec.strip()
		uid, mid, rating, timestamp = rec.split('::')
		uid = int(uid) - 1
		mid = int(mid) - 1
		rating = int(rating)

		A[uid, mid] = rating

	if binary: A = (A>=3).astype(np.float32)
	return A

def load_movielen_10m(dir='.', sample_ratio=1, rd_seed=None, binary=True):
	# for reproducibility
	if rd_seed is not None:
		np.random.seed(rd_seed)

	m, n = 71567, 10681  # #user and #movie can be found in readme file
	uids = []; mids = []; ratings = []
	with open(dir+'/ratings.dat', 'r', encoding='utf-8') as f:
		count = 0
		while (line:=f.readline()):
			count += 1
			if count % 100000 == 0: print(f'\rLoading movielen dataset {count/10000000*100:.2f}%...', end='')
			if sample_ratio < 1 and np.random.rand() > sample_ratio: continue
			rec = line.strip()
			uid, mid, rating, timestamp = rec.split('::')
			uid = int(uid) - 1;
			mid = int(mid) - 1;
			rating = float(rating);

			uids.append(uid)
			mids.append(mid)
			ratings.append(rating if not binary else rating >= 3)

			#if count==300: break

	#print(mids)
	all_mids = sorted(list(set(mids))) # mid不连续，得重新处理一下 - 别忘了去重！
	mids = [idx(mid, all_mids) for mid in mids]
	print('Resorting done')

	return sp.csr_matrix((ratings, (uids, mids)), shape=(m, n), dtype=np.float32)

load_movielen_25m = load_movielen

def load_amazon_all_beauty(dir='.', sample_ratio=0.02, rd_seed=None, binary=True) -> sp.csr_matrix:
	if rd_seed is not None:
		np.random.seed(rd_seed)

	filepath = dir+'/amazon_all_beauty/All_Beauty.jsonl'
	records = [] # list of record:=(user id, product id, rating)
	with open(filepath, 'r') as file:
		for line in tqdm(file):
			review = json.loads(line.strip())
			review = (review['user_id'], review['parent_asin'], review['rating'])
			records.append(review)
		records.sort()

	# shall be ordered
	all_users = sorted(list(set([record[0] for record in records])))
	all_products = sorted(list(set([record[1] for record in records])))
	M = len(all_users); N = len(all_products)

	m = math.floor(M*sample_ratio)
	n = math.floor(N*sample_ratio)

	selected_users = sorted(np.random.choice(all_users, m))
	selected_products = sorted(np.random.choice(all_products, n))

	rows = []; cols = []; data = []
	for record in tqdm(records):
		i = idx(record[0], selected_users)
		j = idx(record[1], selected_products)
		if i is None or j is None: continue

		rows.append(i)
		cols.append(j)
		data.append(record[2] if not binary else (record[2]>=3))

	return sp.csr_matrix((data, (rows, cols)), shape=(m, n), dtype=np.float32)



def load_amazon_all_beauty(dir='.', sample_ratio=0.02, rd_seed=None, binary=True) -> sp.csr_matrix:
	if rd_seed is not None:
		np.random.seed(rd_seed)

	filepath = dir+'/amazon_all_beauty/All_Beauty.jsonl'
	records = [] # list of record:=(user id, product id, rating)
	with open(filepath, 'r') as file:
		for line in tqdm(file):
			review = json.loads(line.strip())
			review = (review['user_id'], review['parent_asin'], review['rating'])
			records.append(review)
		records.sort()

	# shall be ordered
	all_users = sorted(list(set([record[0] for record in records])))
	all_products = sorted(list(set([record[1] for record in records])))
	M = len(all_users); N = len(all_products)

	m = math.floor(M*sample_ratio)
	n = math.floor(N*sample_ratio)

	selected_users = sorted(np.random.choice(all_users, m))
	selected_products = sorted(np.random.choice(all_products, n))

	rows = []; cols = []; data = []
	for record in tqdm(records):
		i = idx(record[0], selected_users)
		j = idx(record[1], selected_products)
		if i is None or j is None: continue

		rows.append(i)
		cols.append(j)
		data.append(record[2] if not binary else (record[2]>=3))

	return sp.csr_matrix((data, (rows, cols)), shape=(m, n), dtype=np.float32)