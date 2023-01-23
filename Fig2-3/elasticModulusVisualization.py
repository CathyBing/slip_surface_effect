#!/usr/bin/env python3

# data structure for unitsphere triangulation
# global array "connectivity" holding three node indices per triangle
# global array "node" containing the 3D coordinates
# global hash "nodeChild" with key made from both parents and child node index as value

import sys
import os
import argparse
import numpy as np
import vtk
import colormaps


node = np.array( [ 
    [ 1.0, 0.0, 0.0], # 0 
    [-1.0, 0.0, 0.0], # 1
    [ 0.0, 1.0, 0.0], # 2 
    [ 0.0,-1.0, 0.0], # 3
    [ 0.0, 0.0, 1.0], # 4 
    [ 0.0, 0.0,-1.0]  # 5                                
    ] )
    
octahedron = np.array( [ 
    [ 0, 2, 4 ],
    [ 2, 1, 4 ],
    [ 1, 3, 4 ],
    [ 3, 0, 4 ],
    [ 0, 5, 2 ],
    [ 2, 5, 1 ],
    [ 1, 5, 3 ],
    [ 3, 5, 0 ],
    ] )


def iszero(a):
  return np.isclose(a,0.0,atol=1.0e-300,rtol=0.0)

def isone(a):
  return np.isclose(a,1.0,atol=1.0e-15,rtol=0.0)


def om2ax(om):
  """Orientation matrix to axis angle"""
  P=-1
  ax=np.empty(4)

  # first get the rotation angle
  t = 0.5*(om.trace()-1.0)
  ax[3] = np.arccos(np.clip(t,-1.0,1.0))
  
  if iszero(ax[3]):
    ax = [ 0.0, 0.0, 1.0, 0.0]
  else:
    w,vr = np.linalg.eig(om)
  # next, find the eigenvalue (1,0j)
    i = np.where(np.isclose(w,1.0+0.0j))[0][0]
    ax[0:3] = np.real(vr[0:3,i])
    diagDelta = np.array([om[1,2]-om[2,1],om[2,0]-om[0,2],om[0,1]-om[1,0]])
    ax[0:3] = np.where(iszero(diagDelta), ax[0:3],np.abs(ax[0:3])*np.sign(-P*diagDelta))
  
  return np.array(ax)


def inverse66(M66):
    """ideas from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.622.4732&rep=rep1&type=pdf"""
    W = np.identity(6)
    W[3,3] = W[4,4] = W[5,5] = 0.5
    return np.einsum('ij,jk,kl',W,np.linalg.inv(M66),W)


def C66toC3333(stiffness):
    index = np.array([[0,0],[1,1],[2,2,],[1,2],[0,2],[0,1]])
    C3333 = np.zeros((3,3,3,3))
    for a in range(6):
      i = index[a][0]
      j = index[a][1]
      for b in range(6):
        k = index[b][0]
        l = index[b][1]
        C3333[i,j,k,l] = stiffness[a,b]
        C3333[i,j,l,k] = stiffness[a,b]
        C3333[j,i,k,l] = stiffness[a,b]
        C3333[j,i,l,k] = stiffness[a,b]
        
    return C3333
        

def E_hkl3333(S3333,dir):

    return 1./np.einsum('i,j,k,l,ijkl',dir,dir,dir,dir,S3333)


def SierpinskySpherical(t,N):
    # Subdivide the triangle and normalize
    #  the new points thus generated to lie on the surface of the unit
    #  sphere.
    #  input triangle with vertices labeled [0,1,2] as shown
    #  below will be turned into four new triangles:
    #
    #            Make new (auto normalized) points
    #                 a = (0+1)/2
    #                 b = (1+2)/2
    #                 c = (2+0)/2
    #       C=2
    #       /\
    #      /  \
    #    c/____\ b    Construct new triangles
    #    /\    /\       t1 [0,a,c]
    #   /  \  /  \      t2 [a,1,b]
    #  /____\/____\     t3 [c,b,2]
    # 0=A    a   B=1    t4 [a,b,c] 
    
    if N > 0:
      a = indexOfChild(t[[0,1]])
      b = indexOfChild(t[[1,2]])
      c = indexOfChild(t[[2,0]])

      return np.vstack((
                        SierpinskySpherical(np.array([    t[0],a,c]),N-1),
                        SierpinskySpherical(np.array([  a,t[1],b]),  N-1),
                        SierpinskySpherical(np.array([c,b,t[2]]),    N-1),
                        SierpinskySpherical(np.array([a,b,c]),       N-1),
                      ))
    else:
      return t


def indexOfChild(parents):
    child = '{}+{}'.format(str(np.min(parents)),str(np.max(parents)))
    if child not in nodeChild:
      nodeChild[child] = len(node)                                              # find next highest index
      node.resize(nodeChild[child]+1,3)                                         # make room for new node
      node[nodeChild[child]] = np.average(node[parents],axis=0)                 # average of both parents
      node[nodeChild[child]] /= np.linalg.norm(node[nodeChild[child]])          # normalize to unit sphere

    return nodeChild[child]


def C66fromSymmetry(c11=0.0,c12=0.0,c13=0.0,c14=0.0,c15=0.0,c16=0.0,
                            c22=0.0,c23=0.0,c24=0.0,c25=0.0,c26=0.0,
                                    c33=0.0,c34=0.0,c35=0.0,c36=0.0,
                                            c44=0.0,c45=0.0,c46=0.0,
                                                    c55=0.0,c56=0.0,
                                                            c66=0.0,
                    symmetry=None,
                    ):
    """RFS Hearmon, The Elastic Constants of Anisotropic Materials, Reviews of Modern Physics 18 (1946) 409-440"""
    
    C = np.zeros((6,6),dtype=float)

    if symmetry in ['isotropic','cubic','tetragonal','hexagonal','orthorhombic','monoclinic','triclinic']:
      C[0,0] = C[1,1] = C[2,2] = c11
      C[3,3] = C[4,4] = C[5,5] = 0.5*(c11-c12)

      C[0,1] = C[0,2] = C[1,2] = \
      C[1,0] = C[2,0] = C[2,1] = c12

    if symmetry in ['cubic','tetragonal','hexagonal','orthorhombic','monoclinic','triclinic']:
      C[3,3] = C[4,4] = C[5,5] = c44 if c44 > 0.0 else C[3,3]

    if symmetry in ['tetragonal','hexagonal','orthorhombic','monoclinic','triclinic']:
      C[2,2]                   = c33 if c33 > 0.0 else C[0,0]
      C[5,5]                   = c66 if c66 > 0.0 else C[3,3]

      C[0,2] = C[1,2]          = \
      C[2,0] = C[2,1]          = c13 if c13 > 0.0 else C[0,2]
      C[0,5]                   = c16 if c16 > 0.0 else 0.0
      C[5,0]                   = -c16 if c16 > 0.0 else 0.0

    if symmetry in ['hexagonal','orthorhombic','monoclinic','triclinic']:
      C[5,5]                   = 0.5*(c11-c12)

    if symmetry in ['orthorhombic','monoclinic','triclinic']:
      C[1,1]                   = c22 if c22 > 0.0 else C[0,0]
      C[2,2]                   = c33 if c33 > 0.0 else C[0,0]
      C[4,4]                   = c55 if c55 > 0.0 else C[3,3]
      C[5,5]                   = c66 if c66 > 0.0 else C[3,3]
      C[1,2] = C[2,1]          = c23 if c23 > 0.0 else C[1,2]

    if symmetry in ['monoclinic','triclinic']:
      C[1,5] = C[5,1]          = c26 if c26 > 0.0 else 0.0
      C[2,5] = C[5,2]          = c36 if c36 > 0.0 else 0.0
      C[3,4] = C[4,3]          = c45 if c45 > 0.0 else 0.0

    if symmetry in ['triclinic']:
      C[0,3] = C[3,0]          = c14 if c14 > 0.0 else 0.0
      C[0,4] = C[4,0]          = c15 if c15 > 0.0 else 0.0
      C[1,3] = C[3,1]          = c24 if c24 > 0.0 else 0.0
      C[1,4] = C[4,1]          = c25 if c25 > 0.0 else 0.0
      C[2,3] = C[3,2]          = c34 if c34 > 0.0 else 0.0
      C[2,4] = C[4,2]          = c35 if c35 > 0.0 else 0.0
      C[2,5] = C[5,2]          = c36 if c36 > 0.0 else 0.0
      C[3,5] = C[5,3]          = c46 if c46 > 0.0 else 0.0
      C[4,5] = C[5,4]          = c56 if c56 > 0.0 else 0.0
    
    return C


def vtk_writeData(filename):

  polydata = vtk.vtkPolyData()
  triangles = vtk.vtkCellArray()
  triangle = vtk.vtkTriangle()
  magnitude = vtk.vtkDoubleArray()
  magnitude.SetNumberOfComponents(1)
  magnitude.SetName("E");

  points = vtk.vtkPoints()
  for p in node:
    points.InsertNextPoint(*p)
    magnitude.InsertNextValue(np.linalg.norm(p))
    polydata.GetPointData().AddArray(magnitude)

  for t in connectivity:
    for c in range(3):
        triangle.GetPointIds().SetId(c, t[c])
    triangles.InsertNextCell(triangle)
 
  polydata.SetPoints(points)
  polydata.SetPolys(triangles)
 
  writer = vtk.vtkXMLPolyDataWriter()
  writer.SetFileName(os.path.splitext(filename)[0]+".vtp")
  writer.SetInputData(polydata)
  writer.Write()
  

def x3d_writeData(filename):

  ax = om2ax(np.array([[-1., 1., 0.],
                       [-1.,-1., 2.],
                       [ 1., 1., 1.],
                      ])/np.array([np.sqrt(2.),np.sqrt(6.),np.sqrt(3.)])[:,None])

  auto = np.max(np.linalg.norm(node,axis=1))
  minimum = np.min(np.linalg.norm(node,axis=1))

  m = colormaps.Colormap(predefined=args.colormap)
  if args.invert:
    m = m.invert()
    
  output = [
  """
  <html> 
    <head> 
      <title>Elastic Tensor visualization</title>
      <script type='text/javascript' src='http://www.x3dom.org/download/x3dom.js'> </script> 
      <link rel='stylesheet' type='text/css' href='http://www.x3dom.org/download/x3dom.css'></link> 
    </head> 
    <body> 
      <h1>Elastic Tensor visualization</h1> 
      <p>
      Range goes from {min} to {max}
      </p>
      <x3d width='600px' height='600px'> 
      <scene>
        <viewpoint position='{view} {view} {view}' orientation='{axis[0]} {axis[1]} {axis[2]} {angle}'></viewpoint>
        <transform translation='{scale} 0 0' rotation='0 0 1 1.5708'> 
        <shape> 
          <appearance> 
          <material diffuseColor='1 0 0'></material> 
          </appearance> 
          <cylinder radius='{radius}' height='{height}'></cylinder> 
        </shape> 
        </transform> 
        <transform translation='0 {scale} 0'> 
        <shape> 
          <appearance> 
          <material diffuseColor='0 1 0'></material> 
          </appearance> 
          <cylinder radius='{radius}' height='{height}'></cylinder> 
        </shape> 
        </transform> 
        <transform translation='0 0 {scale}' rotation='1 0 0 1.5708'> 
        <shape> 
          <appearance> 
          <material diffuseColor='0 0 1'></material> 
          </appearance> 
          <cylinder radius='{radius}' height='{height}'></cylinder> 
        </shape> 
        </transform> 

        <shape>
          <appearance>
          <material diffuseColor="0.3 0.6 0.2"
                    ambientIntensity="0.167"
                    shininess="0.17"
                    transparency="0.0"
           />
          </appearance>

          <IndexedFaceSet solid="false"
                          convex="true"
                          colorPerVertex="true"
                          creaseAngle="0.0"
                          coordIndex="
  """.format(min=minimum,max=auto,scale=1.5*auto,view=3*auto,axis=ax[:3],angle=ax[3],radius=auto/50.,height=auto)
  ] + \
  [' '.join(map(str,v))+' -1,' for v in connectivity] + \
  [
  '''
            ">
            <coordinate point="
  '''] + \
  [' '.join(map(str,v)) + ', ' for  v in node] + \
  ['''"></coordinate>
            <color color="
  '''
  ] + \
  ['{} {} {}'.format(*(m.color(fraction=np.linalg.norm(v)/auto).expressAs('RGB').color)) + ', ' for  v in node] + \
  [
  '''"></color>

          </IndexedFaceSet>
        </shape>
      </scene>
    </x3d>
  </body> 
  </html>           
  ''']

  with open(os.path.splitext(filename)[0]+".html","w") as f:
    f.write('\n'.join(output) + '\n')


parser = argparse.ArgumentParser()
parser.add_argument("format", 
                    help="output file format",
                    choices=['vtk','x3d'])
parser.add_argument("name", help="output file name")
parser.add_argument("-c", "--colormap",
                    help="colormap for visualization",
                    choices=colormaps.Colormap().predefined(), default='seaweed')
parser.add_argument("-i", "--invert",
                    help="invert colormap",
                    action="store_true")
parser.add_argument("-N", "--recursion",
                    help="number of recursive refinement steps",
                    type=int, default=5)
parser.add_argument("--symmetry",
                    help="crystal structure symmetry",
                    default='isotropic',
                    choices=['triclinic',
                             'monoclinic',
                             'orthorhombic',
                             'hexagonal',
                             'tetragonal',
                             'cubic',
                             'isotropic',
                            ])
for i in range(6):
  for j in range(i,6):
    parser.add_argument("--c{}{}".format(i+1,j+1), type=float, required=i==0 and j<2)

args = parser.parse_args()

S3333 = C66toC3333(inverse66(C66fromSymmetry(c11 = args.c11,
                                             c12 = args.c12,
                                             c13 = args.c13,
                                             c14 = args.c14,
                                             c15 = args.c15,
                                             c16 = args.c16,
                                             c22 = args.c22,
                                             c23 = args.c23,
                                             c24 = args.c24,
                                             c25 = args.c25,
                                             c26 = args.c26,
                                             c33 = args.c33,
                                             c34 = args.c34,
                                             c35 = args.c35,
                                             c36 = args.c36,
                                             c44 = args.c44,
                                             c45 = args.c45,
                                             c46 = args.c46,
                                             c55 = args.c55,
                                             c56 = args.c56,
                                             c66 = args.c66,
                                             symmetry = args.symmetry,
                                            )))

nodeChild = {}
for i in range(len(node)):
  nodeChild['{clone}+{clone}'.format(clone=str(i))] = i

connectivity = np.vstack([SierpinskySpherical(t,args.recursion) for t in octahedron])

for i,n in enumerate(node):
  node[i] *= E_hkl3333(S3333,n)

{'vtk': vtk_writeData,
 'x3d': x3d_writeData,
}[args.format](args.name)
