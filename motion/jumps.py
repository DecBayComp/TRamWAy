
import pandas

def jumps(df, sort=False):
	def diff(df):
		df = df.sort_values('t')
		t = df['t'][1:]
		df = df.diff()
		df['t'] = t
		return df
	return df.groupby(['n'], sort=False).apply(diff)

