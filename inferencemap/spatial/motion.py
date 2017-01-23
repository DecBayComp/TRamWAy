
import pandas

def jumps_impl0(df, sort=True): # very slow; may soon be deprecated
	def diff(df):
		df = df.sort_values('t')
		t = df['t'][1:]
		df = df.diff()
		df['t'] = t
		return df
	return df.groupby(['n'], sort=False).apply(diff)

def jumps(df, sort=False):
	'''each trajectories should be represented by consecutive rows sorted by time.'''
	if sort:
		return jumps_impl0(df) # not exactly equivalent
		#raise NotImplementedError
	i = 'n'
	xyz = ['x', 'y']
	if 'z' in df.columns:
		xyz.append('z')
	ixyz = xyz + [i]
	jump = df[ixyz].diff()
	#df[xyz] = jump[xyz]
	#df = df[jump[i] != 0]
	#return df
	jump = jump[jump[i] == 0][xyz]
	return jump#np.sqrt(np.sum(jump * jump, axis=1))

