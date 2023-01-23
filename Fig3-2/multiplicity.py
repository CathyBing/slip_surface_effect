#!/egr/research/CMM/opt/anaconda3/bin/python

import numpy as np
import damask
import yaml
import os
import sys

# N: int, number of (initial) grains
N = 400
# g: int, grid scale, i.e. total grid number will be g*g*g
g = 96
# cases: int, number of individual random cases, to compare/decrease case-by-case differences.
c = int(sys.argv[1])
# Svar: int, which stiffness shape variation we are using
Svar = 0
# Cvar: int, which CRSS variation we are using
Cvar = 0
# Mvar: multiplicity
Mvar = int(sys.argv[2])

#stiffness shape variation
C = [[110e9, 10e9, 5e9, 130e9, 70e9],
     [165e9, 90e9, 60e9, 190e9, 40e9],
     [165e9, 5e9, 55e9, 140e9, 60e9]]

#CRSS variations
xi_0 = [[200e6, 200e6, 0, 100e6],
        [100e6, 100e6, 0, 100e6],
        [100e6, 200e6, 0, 200e6],
        [100e6, 200e6, 0, 400e6],
        [100e6, 100e6, 0, 200e6],
        [100e6, 100e6, 0, 400e6],
        [100e6, 100e6, 0, 800e6],
        [100e6, 400e6, 0, 400e6]
        ]

#kinematic completeness
N_sl = [3, 3, 0, Mvar]

#create the numerics file
with open('numerics.yaml','w') as num:
    yaml.dump({'grid': {'itmax': 40,
                        'derivative': 'FWBW_difference'}}, num)

#generate grain structure using Voronoi tessellation
geom = damask.Grid.from_Voronoi_tessellation(np.ones(3,dtype=int)*g,
                                            np.ones(3),
                                            np.random.random((N,3)))

#to study the difference of the slip response of near-surface and in-bulk grains, add a layer of air to make a gap
geom_air = geom.canvas(cells = geom.cells+np.array([0,20,0]))

#save
geom_air.save(f'{N}_{g}_film_{c}')

#create material configuration file
with open('material.yaml','w') as cfg:
    '''
    homogenization: no homogenization is needed for a fine grid like this
    microstructure: grain constituents information, orientation etc
    phase: more information about material type
    '''
    #homogeization
    homo = {'homogenization':{'SX':{'N_constituents':1,
                                    'mechanical':{'type': 'pass'}}}}
    yaml.dump(homo, cfg)

    #microstructure
    microlist = []
    for i in range(N):
        microlist.append({
                        'homogenization':'SX',
                        'constituents':[{'v':1.0,
                                        'O':damask.Rotation.from_random().as_quaternion().tolist(),
                                        'phase':'Ti-alpha'}],
                        })
    microlist.append({
                    'homogenization':'SX',
                    'constituents':[{'v':1.0,
                                    'O':[1.0, 0.0, 0.0, 0.0],
                                    'phase':'Air'}],
                    })
    yaml.dump({'material':microlist}, cfg)

    #phase
    phasedict = {
                'Air': {
                        'lattice':'cF', #isotropic
                        'mechanical':{
                                    'output': ['F', 'P', 'F_e', 'F_p', 'L_p'],
                                    'elastic':{'C_11': 1e8, 'C_12': 1e6, 'C_44': 49.5e6, 'type': 'Hooke'},
                                    'plastic':{
                                                'dot_gamma_0': 0.001, #gdot0
                                                'h_0': 1e6, #h0
                                                #'h': 1,
                                                'xi_0': 0.03e6, #tau0
                                                'xi_inf': 0.06e6, #tausat
                                                'n': 5, #n
                                                'M': 3, #m
                                                'a': 2, #a
                                                'output': ['xi'], #resistance against plastic flow, Pa
                                                'dilatation': True,
                                                'type': 'isotropic',
                                                },
                                        },
                        },
                'Ti-alpha': {
                            'lattice':'hP',
                            'c/a': 1.587,
                            'mechanical':{
                                        'output': ['F', 'P', 'F_e', 'F_p', 'L_p', 'O'],
                                        'elastic':{'C_11': C[Svar][0], 'C_12': C[Svar][1], 'C_13': C[Svar][2],
                                                'C_33': C[Svar][3], 'C_44': C[Svar][4], 'type': 'Hooke'},
                                        'plastic':{
                                                    'N_sl': N_sl, #slip family numbers array
                                                    'a_sl': 2.0,
                                                    'dot_gamma_0_sl': 0.001, #gdot0_slip
                                                    'h_0_sl-sl': 200e6, #h0_slipslip
                                                    'h_sl-sl': [+1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, -1.0, -1.0,
                                                      -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  1.0,
                                                      +1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
                                                      +1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
                                                      +1.0,  1.0,  1.0,  1.0,  1.0,  1.0], #interaction_slipslip
                                                    'n_sl': 20,
                                                    'output': ['gamma_sl'], #plastic shear, accumulatedshear_slip
                                                    'xi_0_sl': xi_0[Cvar], #tau0_slip, corresponding to N_sl
                                                    'xi_inf_sl': [x*3 for x in xi_0[Cvar]], #tausat_slip, corresponding to N_sl
                                                    'type': 'phenopowerlaw',
                                                    }
                                            },
                            },
                }
    yaml.dump({'phase':phasedict}, cfg)

#create the film loading file
with open('tensionX.yaml','w') as tsX:
    '''
    fdot: deformation gradiant rate, component-wise exclusive with p
    p: Piola-Kirchhoff stress, component-wise exclusive with fdot
    t: total time
    N: number of time icrements, linear
    '''
    tslist = [{'boundary_conditions': {'mechanical':{'F': [[1.05, 0, 0],
                                                        [0, 1, 0],
                                                        [0, 0, 'x']],
                                                    'P': [['x','x','x'],
                                                        ['x','x','x'],
                                                        ['x','x',0]]}},
                'discretization': {'t': 50,
                                'N': 50},
                'f_out': 50
            }
            ]
    yaml.dump({'solver': {'mechanical': 'spectral_basic'}, 'loadstep': tslist}, tsX)

#film simulation
os.system(f'time mpirun -n 16 DAMASK_grid --load tensionX.yaml --geom {N}_{g}_film_{c}.vti > {N}_{g}_film_{c}.out')

#get result from hdf5 files
result = damask.Result(f'{N}_{g}_film_{c}_tensionX.hdf5')
np.savetxt(f'{N}_{g}_film_{c}_slip.txt', result.place('gamma_sl')[f'increment_50'].__array__(), fmt='%.4e')
os.system(f'rm {N}_{g}_film_{c}_tensionX.hdf5')
