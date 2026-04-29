import bisect
import math

import numpy as np
import scipy.sparse as sp

import csv, json
from tqdm import tqdm

def idx(val, arr): # find val in arr, based on binary search
	res = bisect.bisect_left(arr, val)
	if res == len(arr) or arr[res] != val: return None
	return res

#* Different versions of MovieLens have different formats;
#* So we need to prepare different loader functions
#* `split` versions load dataset into 80/20 train/test datasets

# load MovieLen-latest-small or MovieLen-25m
def load_movielen(dir='.', sample_ratio=1, rd_seed=None, binary=True, threshold=3) -> sp.csr_matrix:
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
			data.append(rating if not binary else (rating>=threshold))

	A = sp.csr_matrix((data, (rows, cols)), shape=(m, n), dtype=np.float32)
	print('\rLoading done!', end='')

	return A

def load_movielen_split(dir='.', sample_ratio=1, rd_seed=None, binary=True, threshold=3.5, train_ratio=0.8):
	with open(dir+'/manifest.json', 'r', encoding='utf-8') as f:
		metadata = json.load(f)
	nr_movies = metadata['nr_movies']
	nr_users  = metadata['nr_users']
	nr_records= metadata['nr_records']

	if rd_seed is not None:
		np.random.seed(rd_seed)

	sample_ratio = min(max(sample_ratio, 0), 1)
	train_ratio = min(max(train_ratio, 0), 1)
	m, n = nr_users, nr_movies

	# load movies and their id(mid=movie id)
	with open(dir+'/movies.csv', 'r', encoding='utf-8') as f:
		reader = csv.reader(f)
		reader.__next__()
		all_mid = np.array([int(row[0]) for row in reader])
	movie_to_col = {int(mid): j for j, mid in enumerate(all_mid)}

	sampled_record_cnt = math.floor(nr_records*sample_ratio)
	train_record_cnt = math.floor(sampled_record_cnt*train_ratio)
	if sampled_record_cnt == 0:
		A_train = sp.csr_matrix((m, n), dtype=np.float32)
		A_test = sp.csr_matrix((m, n), dtype=np.float32)
		Ind_train = [[] for _ in range(m)]
		Ind_test = [[] for _ in range(m)]
		return A_train, A_test, Ind_train, Ind_test

	sample_mask = np.zeros(nr_records, dtype=bool)
	if sampled_record_cnt == nr_records:
		sample_mask[:] = True
	elif sampled_record_cnt > 0:
		sample_mask[np.random.choice(nr_records, sampled_record_cnt, replace=False)] = True

	train_mask = np.zeros(sampled_record_cnt, dtype=bool)
	if train_record_cnt > 0:
		train_mask[np.random.choice(sampled_record_cnt, train_record_cnt, replace=False)] = True

	rows_train = []; cols_train = []; data_train = []
	rows_test = []; cols_test = []; data_test = []
	Ind_train = [[] for _ in range(m)]
	Ind_test = [[] for _ in range(m)]

	low_mask = (1<<16)-1
	with open(dir+'/ratings.csv', 'r', encoding='utf-8') as f:
		reader = csv.reader(f)
		reader.__next__() # consume title row
		rec_cnt = 0
		sampled_rec_id = 0
		for row in reader:
			if rec_cnt & low_mask == 0: # per 16384 records - bit ops should be faster than mod
				print(f'\rLoading movielen split {rec_cnt/nr_records*100:.2f}%...', end='')

			if not sample_mask[rec_cnt]:
				rec_cnt += 1
				continue

			uid = int(row[0]) # user id
			mid = int(row[1]) # movie id
			rating = float(row[2])

			i = uid - 1 # uid starts from 1
			j = movie_to_col.get(mid)
			is_train_record = train_mask[sampled_rec_id]
			sampled_rec_id += 1
			if (i < 0) or (i >= m) or (j is None):
				rec_cnt += 1
				continue

			value = rating if not binary else (rating>=threshold)
			if is_train_record:
				rows_train.append(i)
				cols_train.append(j)
				data_train.append(value)
				Ind_train[i].append(j)
			else:
				rows_test.append(i)
				cols_test.append(j)
				data_test.append(value)
				Ind_test[i].append(j)

			rec_cnt += 1

	A_train = sp.csr_matrix((data_train, (rows_train, cols_train)), shape=(m, n), dtype=np.float32)
	A_test = sp.csr_matrix((data_test, (rows_test, cols_test)), shape=(m, n), dtype=np.float32)
	print('\rLoading done!                  ', end='')

	return A_train, A_test, Ind_train, Ind_test

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

def load_movielen_100k_split(dir='.', sample_ratio=1, rd_seed=None, binary=True, threshold=3.5, train_ratio=0.8):
	if rd_seed is not None:
		np.random.seed(rd_seed)

	with open(dir+'/u.data', 'r', encoding='utf-8') as f:
		records = f.readlines()

	m, n = 943, 1682 # #user and #movie can be found in readme file
	nr_records = len(records)
	sample_ratio = min(max(sample_ratio, 0), 1)
	train_ratio = min(max(train_ratio, 0), 1)

	sampled_record_cnt = math.floor(nr_records*sample_ratio)
	train_record_cnt = math.floor(sampled_record_cnt*train_ratio)
	if sampled_record_cnt == 0:
		A_train = sp.csr_matrix((m, n), dtype=np.float32)
		A_test = sp.csr_matrix((m, n), dtype=np.float32)
		Ind_train = [[] for _ in range(m)]
		Ind_test = [[] for _ in range(m)]
		return A_train, A_test, Ind_train, Ind_test

	sample_mask = np.zeros(nr_records, dtype=bool)
	if sampled_record_cnt == nr_records:
		sample_mask[:] = True
	else:
		sample_mask[np.random.choice(nr_records, sampled_record_cnt, replace=False)] = True

	train_mask = np.zeros(sampled_record_cnt, dtype=bool)
	if train_record_cnt > 0:
		train_mask[np.random.choice(sampled_record_cnt, train_record_cnt, replace=False)] = True

	rows_train = []; cols_train = []; data_train = []
	rows_test = []; cols_test = []; data_test = []
	Ind_train = [[] for _ in range(m)]
	Ind_test = [[] for _ in range(m)]

	low_mask = (1<<14)-1
	sampled_rec_id = 0
	for rec_id, rec in enumerate(records):
		if rec_id & low_mask == 0:
			print(f'\rLoading movielen 100k split {rec_id/nr_records*100:.2f}%...', end='')

		if not sample_mask[rec_id]:
			continue

		uid, mid, rating, timestamp = rec.strip().split('\t')
		i = int(uid) - 1
		j = int(mid) - 1
		rating = int(rating)

		is_train_record = train_mask[sampled_rec_id]
		sampled_rec_id += 1

		value = rating if not binary else (rating>=threshold)
		if is_train_record:
			rows_train.append(i)
			cols_train.append(j)
			data_train.append(value)
			Ind_train[i].append(j)
		else:
			rows_test.append(i)
			cols_test.append(j)
			data_test.append(value)
			Ind_test[i].append(j)

	A_train = sp.csr_matrix((data_train, (rows_train, cols_train)), shape=(m, n), dtype=np.float32)
	A_test = sp.csr_matrix((data_test, (rows_test, cols_test)), shape=(m, n), dtype=np.float32)
	print('\rLoading done!                       ', end='')

	return A_train, A_test, Ind_train, Ind_test

def _movielen_dat_movie_to_col(dir, nr_movies):
	filepath = dir+'/movies.dat'
	if not os.path.exists(filepath):
		return {mid: mid-1 for mid in range(1, nr_movies+1)}

	with open(filepath, 'r', encoding='latin-1') as f:
		all_mids = sorted([int(line.split('::', 1)[0]) for line in f])
	return {mid: j for j, mid in enumerate(all_mids)}

def _load_movielen_dat_split(
	dir='.',
	nr_users=0,
	nr_movies=0,
	nr_records=0,
	dataset_name='movielen',
	sample_ratio=1,
	rd_seed=None,
	binary=True,
	threshold=3.5,
	train_ratio=0.8,
):
	if rd_seed is not None:
		np.random.seed(rd_seed)

	sample_ratio = min(max(sample_ratio, 0), 1)
	train_ratio = min(max(train_ratio, 0), 1)
	m, n = nr_users, nr_movies
	movie_to_col = _movielen_dat_movie_to_col(dir, nr_movies)

	sampled_record_cnt = math.floor(nr_records*sample_ratio)
	train_record_cnt = math.floor(sampled_record_cnt*train_ratio)
	if sampled_record_cnt == 0:
		A_train = sp.csr_matrix((m, n), dtype=np.float32)
		A_test = sp.csr_matrix((m, n), dtype=np.float32)
		Ind_train = [[] for _ in range(m)]
		Ind_test = [[] for _ in range(m)]
		return A_train, A_test, Ind_train, Ind_test

	sample_mask = np.zeros(nr_records, dtype=bool)
	if sampled_record_cnt == nr_records:
		sample_mask[:] = True
	else:
		sample_mask[np.random.choice(nr_records, sampled_record_cnt, replace=False)] = True

	train_mask = np.zeros(sampled_record_cnt, dtype=bool)
	if train_record_cnt > 0:
		train_mask[np.random.choice(sampled_record_cnt, train_record_cnt, replace=False)] = True

	rows_train = []; cols_train = []; data_train = []
	rows_test = []; cols_test = []; data_test = []
	Ind_train = [[] for _ in range(m)]
	Ind_test = [[] for _ in range(m)]

	low_mask = (1<<16)-1
	with open(dir+'/ratings.dat', 'r', encoding='utf-8') as f:
		rec_cnt = 0
		sampled_rec_id = 0
		while (line:=f.readline()):
			if rec_cnt & low_mask == 0:
				print(f'\rLoading {dataset_name} split {rec_cnt/nr_records*100:.2f}%...', end='')

			if not sample_mask[rec_cnt]:
				rec_cnt += 1
				continue

			uid, mid, rating, timestamp = line.strip().split('::')
			i = int(uid) - 1
			j = movie_to_col.get(int(mid))
			rating = float(rating)

			is_train_record = train_mask[sampled_rec_id]
			sampled_rec_id += 1
			if (i < 0) or (i >= m) or (j is None):
				rec_cnt += 1
				continue

			value = rating if not binary else (rating>=threshold)
			if is_train_record:
				rows_train.append(i)
				cols_train.append(j)
				data_train.append(value)
				Ind_train[i].append(j)
			else:
				rows_test.append(i)
				cols_test.append(j)
				data_test.append(value)
				Ind_test[i].append(j)

			rec_cnt += 1

	if rec_cnt != nr_records:
		raise ValueError(f'{dir}/ratings.dat contains {rec_cnt} records, expected {nr_records}')

	A_train = sp.csr_matrix((data_train, (rows_train, cols_train)), shape=(m, n), dtype=np.float32)
	A_test = sp.csr_matrix((data_test, (rows_test, cols_test)), shape=(m, n), dtype=np.float32)
	print('\rLoading done!                  ', end='')

	return A_train, A_test, Ind_train, Ind_test

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

def load_movielen_1m_split(dir='.', sample_ratio=1, rd_seed=None, binary=True, threshold=3.5, train_ratio=0.8):
	return _load_movielen_dat_split(
		dir=dir,
		nr_users=6040,
		nr_movies=3952,
		nr_records=1000209,
		dataset_name='movielen 1m',
		sample_ratio=sample_ratio,
		rd_seed=rd_seed,
		binary=binary,
		threshold=threshold,
		train_ratio=train_ratio,
	)

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
	all_mids = sorted(list(set(mids))) # mid is not necessarily contiguous!
	mids = [idx(mid, all_mids) for mid in mids]
	print('Resorting done')

	return sp.csr_matrix((ratings, (uids, mids)), shape=(m, n), dtype=np.float32)

def load_movielen_10m_split(dir='.', sample_ratio=1, rd_seed=None, binary=True, threshold=3.5, train_ratio=0.8):
	return _load_movielen_dat_split(
		dir=dir,
		nr_users=71567,
		nr_movies=10681,
		nr_records=10000054,
		dataset_name='movielen 10m',
		sample_ratio=sample_ratio,
		rd_seed=rd_seed,
		binary=binary,
		threshold=threshold,
		train_ratio=train_ratio,
	)

load_movielen_25m = load_movielen
load_movielen_25m_split = load_movielen_split

def _netflix_files(dir):
	data_files = [os.path.join(dir, f'combined_data_{i}.txt') for i in range(1, 5)]
	if all(os.path.exists(filepath) for filepath in data_files):
		return data_files

	nested_dir = os.path.join(dir, 'Netflix')
	data_files = [os.path.join(nested_dir, f'combined_data_{i}.txt') for i in range(1, 5)]
	if all(os.path.exists(filepath) for filepath in data_files):
		return data_files

	raise FileNotFoundError('Cannot find Netflix combined_data_1.txt ... combined_data_4.txt')

def _count_netflix_records(data_files):
	total = 0
	chunk_size = 16 * 1024 * 1024
	for filepath in data_files:
		comma_count = 0
		with open(filepath, 'rb') as f:
			while chunk := f.read(chunk_size):
				comma_count += chunk.count(b',')

		if comma_count % 2 != 0:
			raise ValueError(f'{filepath} has an odd comma count; unexpected Netflix record format')
		total += comma_count // 2

	return total

def load_netflix(dir='Netflix', sample_ratio=1, rd_seed=None, binary=True) -> sp.csr_matrix:
	# Netflix Prize training data:
	# MovieIDs are 1..17770; CustomerIDs have gaps and need remapping.
	nr_movies = 17770
	nr_records = 100480507
	data_files = _netflix_files(dir)

	if rd_seed is not None:
		np.random.seed(rd_seed)

	if sample_ratio <= 0:
		return sp.csr_matrix((0, 0), dtype=np.float32)

	if sample_ratio >= 1:
		selected_movies = list(range(1, nr_movies+1))
	else:
		selected_movies = sorted(
			np.random.choice(np.arange(1, nr_movies+1), math.floor(nr_movies*sample_ratio), replace=False)
		)
	selected_movie_to_col = {mid: j for j, mid in enumerate(selected_movies)}

	# First pass: collect real customer ids. They range up to 2649429 with gaps,
	# so compact row indices require discovering the ids in the raw files.
	all_users = set()
	low_mask = (1<<18)-1
	rec_cnt = 0
	for filepath in data_files:
		with open(filepath, 'r', encoding='utf-8') as f:
			for line in f:
				if line.endswith(':\n') or line.endswith(':\r\n') or line.endswith(':'):
					continue

				rec_cnt += 1
				if rec_cnt & low_mask == 0:
					print(f'\rScanning netflix users {rec_cnt/nr_records*100:.2f}%...', end='')

				uid = int(line.split(',', 1)[0])
				all_users.add(uid)

	print('\rScanning netflix users done!       ')

	if sample_ratio >= 1:
		selected_users = sorted(all_users)
	else:
		selected_users = sorted(
			np.random.choice(np.array(list(all_users)), math.floor(len(all_users)*sample_ratio), replace=False)
		)
	selected_user_to_row = {uid: i for i, uid in enumerate(selected_users)}

	rows = []; cols = []; data = []
	rec_cnt = 0
	for filepath in data_files:
		current_col = None
		with open(filepath, 'r', encoding='utf-8') as f:
			for line in f:
				if line.endswith(':\n') or line.endswith(':\r\n') or line.endswith(':'):
					mid = int(line.split(':', 1)[0])
					current_col = selected_movie_to_col.get(mid)
					continue

				rec_cnt += 1
				if rec_cnt & low_mask == 0:
					print(f'\rLoading netflix dataset {rec_cnt/nr_records*100:.2f}%...', end='')

				if current_col is None:
					continue

				uid, rating, _ = line.split(',', 2)
				row = selected_user_to_row.get(int(uid))
				if row is None:
					continue

				rows.append(row)
				cols.append(current_col)
				rating = int(rating)
				data.append(rating if not binary else (rating>=3))

	print('\rLoading done!                     ', end='')
	return sp.csr_matrix((data, (rows, cols)), shape=(len(selected_users), len(selected_movies)), dtype=np.float32)

def load_netflix_split(dir='Netflix', sample_ratio=0.02, rd_seed=None, binary=True, threshold=3.5, train_ratio=0.8):
	# Netflix Prize training data:
	# MovieIDs are 1..17770; CustomerIDs have gaps and need remapping.
	nr_movies = 17770
	data_files = _netflix_files(dir)

	if rd_seed is not None:
		np.random.seed(rd_seed)

	sample_ratio = min(max(sample_ratio, 0), 1)
	train_ratio = min(max(train_ratio, 0), 1)
	if sample_ratio == 0:
		A_train = sp.csr_matrix((0, nr_movies), dtype=np.float32)
		A_test = sp.csr_matrix((0, nr_movies), dtype=np.float32)
		return A_train, A_test, [], []

	nr_records = _count_netflix_records(data_files)
	sampled_record_cnt = math.floor(nr_records*sample_ratio)
	train_record_cnt = math.floor(sampled_record_cnt*train_ratio)
	if sampled_record_cnt == 0:
		A_train = sp.csr_matrix((0, nr_movies), dtype=np.float32)
		A_test = sp.csr_matrix((0, nr_movies), dtype=np.float32)
		return A_train, A_test, [], []

	py_rng = random.Random(rd_seed)
	if sampled_record_cnt == nr_records:
		sampled_record_ids = None
	else:
		sampled_record_ids = sorted(py_rng.sample(range(nr_records), sampled_record_cnt))

	train_mask = np.zeros(sampled_record_cnt, dtype=bool)
	if train_record_cnt > 0:
		train_mask[py_rng.sample(range(sampled_record_cnt), train_record_cnt)] = True

	def is_movie_line(line):
		return line.endswith(':\n') or line.endswith(':\r\n') or line.endswith(':')

	# First pass: collect users appearing in sampled records.
	all_users = set()
	low_mask = (1<<18)-1
	rec_cnt = 0
	sampled_ptr = 0
	for filepath in data_files:
		with open(filepath, 'r', encoding='utf-8') as f:
			for line in f:
				if is_movie_line(line):
					continue

				if rec_cnt & low_mask == 0:
					print(f'\rScanning netflix split users {rec_cnt/nr_records*100:.2f}%...', end='')

				if sampled_record_ids is None:
					is_sampled = True
				else:
					is_sampled = sampled_ptr < sampled_record_cnt and rec_cnt == sampled_record_ids[sampled_ptr]
					if is_sampled:
						sampled_ptr += 1

				if is_sampled:
					uid = int(line.split(',', 1)[0])
					all_users.add(uid)

				rec_cnt += 1

	selected_users = sorted(all_users)
	selected_user_to_row = {uid: i for i, uid in enumerate(selected_users)}
	m, n = len(selected_users), nr_movies

	rows_train = []; cols_train = []; data_train = []
	rows_test = []; cols_test = []; data_test = []
	Ind_train = [[] for _ in range(m)]
	Ind_test = [[] for _ in range(m)]

	rec_cnt = 0
	sampled_ptr = 0
	sampled_rec_id = 0
	for filepath in data_files:
		current_col = None
		with open(filepath, 'r', encoding='utf-8') as f:
			for line in f:
				if is_movie_line(line):
					mid = int(line.split(':', 1)[0])
					current_col = mid - 1
					continue

				if rec_cnt & low_mask == 0:
					print(f'\rLoading netflix split {rec_cnt/nr_records*100:.2f}%...', end='')

				if sampled_record_ids is None:
					is_sampled = True
				else:
					is_sampled = sampled_ptr < sampled_record_cnt and rec_cnt == sampled_record_ids[sampled_ptr]
					if is_sampled:
						sampled_ptr += 1

				if not is_sampled:
					rec_cnt += 1
					continue

				uid, rating, _ = line.split(',', 2)
				row = selected_user_to_row.get(int(uid))
				is_train_record = train_mask[sampled_rec_id]
				sampled_rec_id += 1
				if (row is None) or (current_col is None):
					rec_cnt += 1
					continue

				rating = int(rating)
				value = rating if not binary else (rating>=threshold)
				if is_train_record:
					rows_train.append(row)
					cols_train.append(current_col)
					data_train.append(value)
					Ind_train[row].append(current_col)
				else:
					rows_test.append(row)
					cols_test.append(current_col)
					data_test.append(value)
					Ind_test[row].append(current_col)

				rec_cnt += 1

	A_train = sp.csr_matrix((data_train, (rows_train, cols_train)), shape=(m, n), dtype=np.float32)
	A_test = sp.csr_matrix((data_test, (rows_test, cols_test)), shape=(m, n), dtype=np.float32)
	print('\rLoading done!                  ', end='')

	return A_train, A_test, Ind_train, Ind_test

load_movielen_25m = load_movielen
load_movielen_25m_split = load_movielen_split