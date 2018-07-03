
import os.path
import sys
from tramway.core import *
import tramway.core.analyses.base as ba
from tramway.core.hdf5 import *
from tramway.helper.tessellation import *
from tramway.helper.inference import *
xyt = load_xyt('trajectories_2.txt')
#xyt = load_xyt('~/github/TRamWAy/tests/VLP/WT/1/folder_130219_24h_C01/trajectories_13.txt')
#xyt = load_xyt('~/tmp/trajectories_18.txt')
time =np.arange(0., 1601, 200.)
frames = np.c_[time[:-2], time[2:]]
print(frames)
xyt_ = xyt[(6.75<xyt['x'])&(xyt['x']<7.35)&(14.25<xyt['y'])&(xyt['y']<14.85)]
#xyt_ = xyt[(7.2<xyt['x'])&(xyt['x']<7.8)&(13.2<xyt['y'])&(xyt['y']<13.8)]
#xyt_ = xyt[(6.7<xyt['x'])&(xyt['x']<7.1)&(13.<xyt['y'])&(xyt['y']<13.4)]
#xyt_ = xyt[(4.2<xyt['x'])&(xyt['x']<4.7)&(6.75<xyt['y'])&(xyt['y']<7.25)]
#xyt_ = xyt[(15.1<xyt['x'])&(xyt['x']<15.5)&(4.5<xyt['y'])&(xyt['y']<4.9)]
#xyt_ = xyt[(9.9<xyt['x'])&(xyt['x']<10.3)&(12.2<xyt['y'])&(xyt['y']<12.6)]
#xyt_ = xyt[(xyt['x']>15.75)&(xyt['x']<16.25)&(xyt['y']>4.4)&(xyt['y']<4.9)]
xyt_.index = np.arange(xyt_.shape[0])
cells = tessellate(xyt_, 'hexagon', avg_cell_count=10)
i = 0
for t1,t2 in frames:
        print((i, t1,t2))
        rwa_file = 'vlp_2_2_t'+str(i)+'.rwa'
        #if os.path.isfile(rwa_file):
        #        i += 1
        #        continue
        I = np.logical_and(xyt_['t']>t1, xyt_['t']<t2)
        _xyt = xyt_[I].copy()
        _xyt.index = np.arange(_xyt.shape[0])
        #_xyt.to_csv('vlp_1_18_t'+str(i)+'.txt', sep="\t", header=False)
        cells.points = _xyt
        cells.cell_index = cells.tessellation.cell_index(_xyt, knn=50)
        _map = infer(cells, 'stochastic.dv', potential_prior=1., max_iter=1000, verbose=True)
        _a = Analyses(xyt_)
        _c = Analyses(cells)
        _c.add(Maps(_map, mode='stochastic.dv'), label='dv')
        _a.add(_c, label='hexagon')
        save_rwa(rwa_file, _a, force=True)
        i += 1
sys.exit()

cells = tessellate(xyt_, 'hexagon', avg_cell_count=50)
n = cells.tessellation._cell_centers.shape[0]
dynamic_cells = with_time_lattice(cells, frames)
dix = []
i =0
for t1,t2 in frames:
        I = np.logical_and(xyt_['t']>t1, xyt_['t']<t2)
        _xyt = xyt_[I]
        _ix = cells.tessellation.cell_index(xyt_, knn=50)
        dix.append((_ix[0], i*n+_ix[1]))
        i+=1
I,J=zip(*dix)
I,J=np.concatenate(I),np.concatenate(J)
#dynamic_cells.cell_index=(I,J)
dynamic_cells.cell_index=sparse.csc_matrix((np.ones(I.size, dtype=bool), (I,J)), shape=(xyt_.shape[0], frames.shape[0]*n))
_map = infer(dynamic_cells, 'stochastic.dv', potential_prior=1., max_iter=10000)

