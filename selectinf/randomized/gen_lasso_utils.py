import numpy as np
from scipy.linalg import svd
import networkx as nx


def create_penalty_matrix(matrix_type,p,k=None,fused_dims=None):
	"""
	create a penalty matrix according to the specified parameters
	----------
	matrix_type : string
		One of ['lasso','fused','trendfiltering'], specifying the
		type of the penalty matrix
	p : int
		the number of features
	k : [int, np.ndarray]
		This parametrizes the construction of the penalty matrix D 
		(k-D fused or k-th order polynomial trend filtering), If None, 
		defaults to 1D fused lasso or linear trend filtering.
	fused_dims : sequence
		Can't be None if penalty_matrix is "fused" and k is an int > 1,
		must be a sequence of length k specifying the dimensions of the grid.
	"""
	if matrix_type == "lasso":
		## maybe should change to just call plain lasso?
		return np.eye(p)
	elif matrix_type == "fused":
		return create_fused_lasso_matrix(p,k,fused_dims)
	elif matrix_type == "trendfiltering":
		return create_trendfiltering_matrix(p,k)

def create_trendfiltering_matrix(p,k=None):
	"""
	p : int
		the number of features
	k : [int, np.ndarray]
		This parametrizes the construction of the penalty matrix D 
		(k-th order polynomial trend filtering), If None, defaults 
		to linear trend filtering.
	"""
	if k is None:
		k = 1
	if isinstance(k,int):
		if k == 0:
			return create_fused_lasso_matrix(p)
		elif k == 1:
			D = -np.eye(p)[:-2,:]
			D[range(p-2),range(1,p-1)] = 2
			D[range(p-2),range(2,p)] = -1
		elif k == 2:
			D = np.eye(p)[:-3,:]
			D[range(p-3),range(1,p-2)] = -3
			D[range(p-3),range(2,p-1)] = 3
			D[range(p-3),range(3,p)] = -1
		elif k == 3:
			D = np.eye(p)[:-4,:]
			D[range(p-4),range(1,p-3)] = -4
			D[range(p-4),range(2,p-2)] = 6
			D[range(p-4),range(3,p-1)] = -4
			D[range(p-4),range(4,p)] = 1
		elif k == 4:
			D = np.eye(p)[:-5,:]
			D[range(p-5),range(1,p-4)] = -5
			D[range(p-5),range(2,p-3)] = 10
			D[range(p-5),range(3,p-2)] = -10
			D[range(p-5),range(4,p-1)] = 5
			D[range(p-5),range(5,p)] = -1
		else:
			raise ValueError("We currently onnly support polynomial trend filtering up to order 4")
		return D
	else:
		raise ValueError("'k' must be either an integer (0,1,2,3,4) or a 2D numpy array")

def create_fused_lasso_matrix(p,k=None,fused_dims=None):
	"""
	p : int
		the number of features
	k : [int, np.ndarray]
		This parametrizes the construction of the penalty matrix D 
		(k-D fused or k-th order polynomial trend filtering) If None, 
		defaults to 1D fused lasso or linear trend filtering.
	fused_dims : sequence
		Can't be None if k is an int > 1, must be a sequence 
		of length k specifying the dimensions of the grid.
	"""
	if k is None:
		k = 1
	if isinstance(k,int):
		D = -np.eye(p)[:-1,:]
		D[range(p-1),range(1,p)] = 1
		if k == 1:
			return D
		if fused_dims is None:
			raise ValueError("If 'k' > 1, fused_dims must be provided")
		elif len(fused_dims) != k:
			raise ValueError("length of 'fused_dims' must equal 'k'")
		else:
			x=fused_dims[0]
			y=fused_dims[1]
			add = np.zeros(((x-1)*y,p))
			add[range(x*y-y),range(y,x*y)] = 1
			D = np.hstack((D,add))
			if k == 2:
				return D
			if k == 3:
				for i in range(2,k+1):
					add = np.zeros(((x-1)*y,p))
					a=(i-1)*x*y-y
					b=i*x*y-y
					add[range(a,b-y),range(a+y,b)] = 1
					D = np.hstack((D,add))

				for i in range(k-1):
					add = np.zeros((x*y,p))
					a = i*x*y
					b = (i+1)*x*y
					c = (i+2)*x*y
					add[range(a,b),range(b,c)] = 1
					D = np.hstack((D,add))

				return D
			else:
				raise ValueError("We only support 'k' = 1,2,3")
	else:
		raise ValueError("'k' must be either an integer (1,2,3) or a 2D numpy array")

def find_trendfiltering_nspaceb(D,active):
	"""
	D : np.ndarray
		penalty matrix
	active : np.ndarray
		boolean vector specifying which rows of D are active (E)
	"""
	k = np.fabs(D[0,1]).astype(int)
	p = D.shape[1]
	if k < 2 or k > 5:
		raise ValueError('k must be an integer in [2,5].')
	nullity = active.sum()
	if nullity == active.shape[0]:
		return np.eye(p)
	D_inactive = D[(~active),:]
	nonzero_idx = np.fabs(D_inactive).sum(axis=0)
	cum_nonzero_idx = nonzero_idx.cumsum()

	# index of first nonzero column in D_{-E}
	if (cum_nonzero_idx==0).any():
		head_offset = np.amax(np.where(cum_nonzero_idx==0)) + 1
	else:
		head_offset = 0

	# index of last nonzerocolumn in D_{-E}
	where_max = np.where(cum_nonzero_idx==np.amax(cum_nonzero_idx))
	tail_offset = len(cum_nonzero_idx) - np.amin(where_max) - 1 

	# where to put initial ones in basis
	active_idx = np.where(active[head_offset:(len(active)-tail_offset)])[0] + head_offset

	mat_dtype = np.longdouble if (k > 2 and p > 50) else np.float64
	basis = np.zeros((p,nullity+k),dtype=mat_dtype)
	head_cols = range(head_offset,nullity-tail_offset)
	basis[(active_idx),head_cols] = 1

	tail_idx = -np.array(range(1,k+1+tail_offset))
	basis[tail_idx,tail_idx] = 1

	basis[range(head_offset),range(head_offset)] = 1

	restr_basis = basis[:,(head_offset):(basis.shape[1]-tail_offset)]
	for r in reversed(range(1+head_offset,p-tail_offset-k+1)):
		if (r-1) in active_idx:
			continue
		else:
			if k == 2: restr_basis[r-1,:] = -restr_basis[r+1,:] + 2*restr_basis[r,:]
			elif k == 3: restr_basis[r-1,:] = restr_basis[r+2,:] - 3*restr_basis[r+1,:] \
											 + 3*restr_basis[r,:]
			elif k == 4: restr_basis[r-1,:] = -restr_basis[r+3,:] + 4*restr_basis[r+2,:] \
											 - 6*restr_basis[r+1,:] + 4*restr_basis[r,:]
			elif k == 5: restr_basis[r-1,:] = restr_basis[r+4,:] - 5*restr_basis[r+3,:] \
											 + 10*restr_basis[r+2,:] - 10*restr_basis[r+1,:] \
											 + 5*restr_basis[r,:]
			
	basis[:,(head_offset):(basis.shape[1]-tail_offset)] = restr_basis
	return basis

def find_fusedlasso_nspaceb(D,active):
	"""
	D : np.ndarray
		penalty matrix
	active : np.ndarray
		boolean vector specifying which rows of D are active (E)
	"""
	p = D.shape[1]
	# create graph from D_{-E}
	D_inactive = D[(~active),:]
	src_nodes = np.where(D_inactive == -1)[1]
	dst_nodes = np.where(D_inactive == 1)[1]
	edge_tuples = [(src_nodes[i],dst_nodes[i]) for i in range(D_inactive.shape[0])]
	
	G = nx.empty_graph(n=p)
	G.add_edges_from(edge_tuples)

	ccs = list(nx.connected_components(G))
	n_ccs = len(ccs)#nx.number_connected_components(G)
	# isos = list(nx.isolates(G))
	# n_isos = len(isos)
	# print("Isolates:", isos)
	basis = np.zeros((p, n_ccs))# + n_isos))

	count = 0
	for i,cc in enumerate(ccs):
		count += len(list(cc))
		# print("~~~~~~")
		# print(list(cc))
		# print(i)
		basis[list(cc),i] = 1
	assert(count==p)
	assert(basis.sum()==p)
	# stop2
	# for i,iso in enumerate(isos):
	# 	basis[iso,i+n_ccs] = 1

	return basis


# In the case of D = I (ordinary lasso), null(D_{-E}) = D_E
def find_nspaceb_identity(active):
	"""
	active : np.ndarray
		boolean vector specifying which rows of D are active (E)
	"""
	return np.identity(len(active))[:,active]

def find_nullspace_basis(D,active):
	"""
	D : np.ndarray
		penalty matrix
	active : np.ndarray
		boolean vector specifying which rows of D are active (E)
	"""
	if (D.shape[0] == D.shape[1]) and (D == np.identity(D.shape[0])).all():
		return find_nspaceb_identity(active)

	_,s,v = svd(D[(~active),:])
	basis = v[len(s):,:].T # v is returned w/ rows being right singular vectors of D_{-E}
	return basis



