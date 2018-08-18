""" Module containing the `~halotools.mock_observables.pair_identification.engines.vdbosch04_pair_finder_engine`
cython function driving the `~halotools.mock_observables.vdbosch04_pair_finder` function.
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
cimport numpy as cnp
cimport cython 
from libc.math cimport ceil, fabs, fmax
from libcpp.vector cimport vector
from libcpp cimport bool



__author__ = ('Johannes Ulf Lange')
__all__ = ('vdbosch04_pair_finder_engine', )

ctypedef bint (*f_type)(cnp.float64_t* w1, cnp.float64_t* w2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def vdbosch04_pair_finder_engine(mesh, x_in, y_in, z_in,
    marks_in, r_h_in, z_h_in, r_s_in, z_s_in, pbcs_in, ell_in, mark_min_in):
    """
    Cython engine for finding pairs according to vdBosch et al. 2004.
    
    Points are considered primaries if
        1. they contain no point with a higher mark within a cylindrical volume
        described by r_h and z_h and
        2. they are not contained within the cylindrical volume given by r_h and
        z_h of another point of higher mark that is considered a central.
    
    Points are considered secondaries if they lie within a cylindrical volume of
    a point that is considered a primary. The volume is defined by r_s and z_s
    of the central. Primaries and their secondaries form a pair.

    Parameters
    ----------
    mesh : object 
        Instance of `~halotools.mock_observables.RectangularMesh`.
    
    x_in : numpy.array
        Array storing Cartesian x-coordinates of the points.
        
    y_in : numpy.array
        Array storing Cartesian y-coordinates of the points.
        
    z_in : numpy.array
        Array storing Cartesian z-coordinates of the points.
        
    marks_in : numpy.ndarray
        Array storing marks for each point in the sample
        
    r_h_in : numpy.array
        Array storing the x-y projected radial distance, radius of cylinder, to
        search for points with higher marks around each point and determine
        wether a point is a primary.
        
    z_h_in : numpy.array
        Array storing the z distance, half the length of a cylinder, to search 
        for points with higher marks around each point and determine whether a
        point is a primary.
        
    r_s_in : numpy.array
        Array storing the x-y projected radial distance, radius of cylinder, to
        search for points each point that will be secondaries to a primary.
        
    z_s_in : numpy.array
        Array storing the z distance, half the length of a cylinder, to search 
        for points each point that will be secondaries to a primary.
        
    pbcs_in : bool
        Boolean describing whether periodic boundary conditions are enforced

    ell_in : float
        Float describing the ellipticity of the cylinder. Instead of using a
        circle in the x-y plane to determine the distance, we use an ellipse.
        Particularly, x^2 / ellipticity^2 + y^2 < r.
    
    mark_min_in : float
        Minimum mark for a point to be considered a primary.
        
    Returns
    -------
    prim : numpy.array
        Int array containing the indices of primaries in primary-secondary
        pairs.

    secd : numpy.array
        Int array containing the indices of secondaries in primary-secondary
        pairs.
    """
    
    cdef cnp.float64_t[:] r_h = np.ascontiguousarray(r_h_in[mesh.idx_sorted])
    cdef cnp.float64_t[:] z_h = np.ascontiguousarray(z_h_in[mesh.idx_sorted])
    cdef cnp.float64_t[:] r_s = np.ascontiguousarray(r_s_in[mesh.idx_sorted])
    cdef cnp.float64_t[:] z_s = np.ascontiguousarray(z_s_in[mesh.idx_sorted])
    cdef cnp.float64_t[:] ell = np.ascontiguousarray(ell_in[mesh.idx_sorted])
    cdef cnp.float64_t mark_min = mark_min_in
    
    cdef cnp.float64_t xperiod = mesh.xperiod
    cdef cnp.float64_t yperiod = mesh.yperiod
    cdef cnp.float64_t zperiod = mesh.zperiod
    cdef bool PBCs = pbcs_in
    
    cdef unsigned int npts = len(x_in)
    cdef cnp.int64_t[:] status = np.zeros(npts, dtype=np.int64)
    cdef vector[cnp.int64_t] prim
    cdef vector[cnp.int64_t] secd
    cdef vector[cnp.int64_t] secd_cand
    
    cdef cnp.float64_t[:] x_all = np.ascontiguousarray(x_in[mesh.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] y_all = np.ascontiguousarray(y_in[mesh.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] z_all = np.ascontiguousarray(z_in[mesh.idx_sorted], dtype=np.float64)
    cdef cnp.float64_t[:] m_all = np.ascontiguousarray(marks_in[mesh.idx_sorted], dtype=np.float64)

    cdef cnp.int64_t cell_index_ctr, cell_index_nbr
    cdef cnp.int64_t[:] cell_indices = np.ascontiguousarray(mesh.cell_id_indices, dtype=np.int64)
    
    cdef cnp.int64_t first_pt, last_pt
    
    cdef int cell_x_nbr, cell_y_nbr, cell_z_nbr, cell_x_ctr, cell_y_ctr, cell_z_ctr
    cdef int nonPBC_cell_x_nbr, nonPBC_cell_y_nbr, nonPBC_cell_z_nbr
    
    cdef int num_cell_x_covering_steps = int(np.ceil(np.max(r_h_in * ell_in) / mesh.xcell_size))
    cdef int num_cell_y_covering_steps = int(np.ceil(np.max(r_h) / mesh.ycell_size))
    cdef int num_cell_z_covering_steps = int(np.ceil(max(np.max(z_h), np.max(z_s)) / mesh.zcell_size))
    
    cdef int leftmost_cell_x_nbr, rightmost_cell_x_nbr
    cdef int leftmost_cell_y_nbr, rightmost_cell_y_nbr
    cdef int leftmost_cell_z_nbr, rightmost_cell_z_nbr
    
    cdef int num_xdivs = mesh.num_xdivs
    cdef int num_ydivs = mesh.num_ydivs
    cdef int num_zdivs = mesh.num_zdivs
    
    cdef cnp.float64_t xcell_size = mesh.xcell_size
    cdef cnp.float64_t ycell_size = mesh.ycell_size
    cdef cnp.float64_t zcell_size = mesh.zcell_size
    
    cdef cnp.float64_t x_ctr, y_ctr, z_ctr, m_ctr
    cdef cnp.float64_t min_xcell, max_xcell, min_ycell, max_ycell, min_zcell, max_zcell
    cdef cnp.float64_t xshift, yshift, zshift, dx, dy, dz, dxy_sq
    cdef cnp.float64_t x_ctr_tmp, y_ctr_tmp, z_ctr_tmp
    cdef cnp.float64_t r_h_ctr, r_h_ctr_sq, z_h_ctr, r_s_ctr, r_s_ctr_sq, z_s_ctr, r_max_ctr_sq, z_max_ctr
    cdef unsigned int Nj, i, j, k
    
    cdef cnp.float64_t[:] x_cell
    cdef cnp.float64_t[:] y_cell
    cdef cnp.float64_t[:] z_cell
    cdef cnp.float64_t[:] m_cell

    cdef cnp.int64_t[:] cells = np.zeros(mesh.npts, dtype=np.int64)
    
    for cell_index_ctr in range(mesh.ncells):
        
        first_pt = cell_indices[cell_index_ctr]
        last_pt = cell_indices[cell_index_ctr+1]
        cells[first_pt:last_pt] = cell_index_ctr
    
    cdef cnp.int64_t[:] pts_sorted = np.argsort(m_all)[::-1]

    for i in pts_sorted:
        if status[i] == 0 and m_all[i] > mark_min:
            
            status[i] = 1
            
            x_ctr = x_all[i]
            y_ctr = y_all[i]
            z_ctr = z_all[i]
            m_ctr = m_all[i]
            r_h_ctr = r_h[i]
            z_h_ctr = z_h[i]
            r_h_ctr_sq = r_h_ctr * r_h_ctr
            r_s_ctr = r_s[i]
            z_s_ctr = z_s[i]
            r_s_ctr_sq = r_s_ctr * r_s_ctr
            r_max_ctr_sq = fmax(r_h_ctr_sq, r_s_ctr_sq)
            z_max_ctr = fmax(z_h_ctr, z_s_ctr)
            
            cell_index_ctr = cells[i]
            cell_x_ctr = cell_index_ctr // (num_ydivs*num_zdivs)
            cell_y_ctr = (cell_index_ctr - cell_x_ctr*num_ydivs*num_zdivs) // num_zdivs
            cell_z_ctr = cell_index_ctr - (cell_x_ctr*num_ydivs*num_zdivs) - (cell_y_ctr*num_zdivs)
            
            leftmost_cell_x_nbr = cell_x_ctr - num_cell_x_covering_steps
            leftmost_cell_y_nbr = cell_y_ctr - num_cell_y_covering_steps
            leftmost_cell_z_nbr = cell_z_ctr - num_cell_z_covering_steps
            
            rightmost_cell_x_nbr = cell_x_ctr + 1 + num_cell_x_covering_steps
            rightmost_cell_y_nbr = cell_y_ctr + 1 + num_cell_y_covering_steps
            rightmost_cell_z_nbr = cell_z_ctr + 1 + num_cell_z_covering_steps
            
            min_xcell = cell_x_ctr * xcell_size
            min_ycell = cell_y_ctr * ycell_size
            min_zcell = cell_z_ctr * zcell_size
            max_xcell = min_xcell + xcell_size
            max_ycell = min_ycell + ycell_size
            max_zcell = min_zcell + zcell_size

            if(x_ctr - min_xcell > r_h_ctr * ell[i]):
                leftmost_cell_x_nbr = leftmost_cell_x_nbr + 1
            if(y_ctr - min_ycell > r_h_ctr):
                leftmost_cell_y_nbr = leftmost_cell_y_nbr + 1
            if(z_ctr - min_zcell > fmax(z_s_ctr, z_h_ctr)):
                leftmost_cell_z_nbr = leftmost_cell_z_nbr + 1
            if(max_xcell - x_ctr > r_h_ctr * ell[i]):
                rightmost_cell_x_nbr = rightmost_cell_x_nbr - 1
            if(max_ycell - y_ctr > r_h_ctr):
                rightmost_cell_y_nbr = rightmost_cell_y_nbr - 1
            if(max_zcell - z_ctr > fmax(z_s_ctr, z_h_ctr)):
                rightmost_cell_z_nbr = rightmost_cell_z_nbr - 1

            for nonPBC_cell_x_nbr in range(leftmost_cell_x_nbr, rightmost_cell_x_nbr):
                if (PBCs) & (nonPBC_cell_x_nbr < 0):
                    xshift = -xperiod
                elif (PBCs) & (nonPBC_cell_x_nbr >= num_xdivs):
                    xshift = +xperiod
                else:
                    xshift = 0.
                # Now apply the PBCs
                cell_x_nbr = nonPBC_cell_x_nbr % num_xdivs
                
                for nonPBC_cell_y_nbr in range(leftmost_cell_y_nbr, rightmost_cell_y_nbr):
                    if (PBCs) & (nonPBC_cell_y_nbr < 0):
                        yshift = -yperiod
                    elif (PBCs) & (nonPBC_cell_y_nbr >= num_ydivs):
                        yshift = +yperiod
                    else:
                        yshift = 0.
                    # Now apply the PBCs
                    cell_y_nbr = nonPBC_cell_y_nbr % num_ydivs
                    
                    for nonPBC_cell_z_nbr in range(leftmost_cell_z_nbr, rightmost_cell_z_nbr):
                        if (PBCs) & (nonPBC_cell_z_nbr < 0):
                            zshift = -zperiod
                        elif (PBCs) & (nonPBC_cell_z_nbr >= num_zdivs):
                            zshift = +zperiod
                        else:
                            zshift = 0.
                        # Now apply the PBCs
                        cell_z_nbr = nonPBC_cell_z_nbr % num_zdivs
                        
                        cell_index_nbr = cell_x_nbr*(num_ydivs*num_zdivs) + cell_y_nbr*num_zdivs + cell_z_nbr
                        first_pt = cell_indices[cell_index_nbr]
                        last_pt = cell_indices[cell_index_nbr+1]
                        Nj = last_pt - first_pt
                        
                        #loop over points in cell1 points
                        if Nj != 0:
                            
                            #extract the points in cell2
                            x_cell = x_all[first_pt:last_pt]
                            y_cell = y_all[first_pt:last_pt]
                            z_cell = z_all[first_pt:last_pt]
                            m_cell = m_all[first_pt:last_pt]
                            
                            x_ctr_tmp = x_ctr - xshift
                            y_ctr_tmp = y_ctr - yshift
                            z_ctr_tmp = z_ctr - zshift
                            
                            #loop over points in cell2 points
                            for j in range(Nj):
                                #calculate the square distance
                                dx = x_ctr_tmp - x_cell[j]
                                dy = y_ctr_tmp - y_cell[j]
                                dz = fabs(z_ctr_tmp - z_cell[j])
                                dxy_sq = dx*dx  / (ell[i] * ell[i]) + dy*dy
                                
                                if (dxy_sq < r_max_ctr_sq) & (dz < z_max_ctr) & (first_pt+j != i):
                                    
                                    if (m_ctr <= m_cell[j]) & (dz < z_h_ctr) & (dxy_sq < r_h_ctr_sq):
                                        status[i] = 2
                                    
                                    if m_ctr > m_cell[j]:
                                        if (dz < z_h_ctr) & (dxy_sq < r_h_ctr_sq):
                                            status[first_pt+j] = 2
                                        if (dxy_sq < r_s_ctr_sq) & (dz < z_s_ctr):
                                            secd_cand.push_back(first_pt+j)
            
            if status[i] == 1:
                for j in range(secd_cand.size()):
                    prim.push_back(i)
                    secd.push_back(secd_cand[j])
                prim.push_back(i)
                secd.push_back(i)
            secd_cand.clear()
    
    # The points we processed above were sorted according to the mesh. We need
    # to undo this sorting operation.
    cdef cnp.int64_t[:] idx_sorted = np.ascontiguousarray(mesh.idx_sorted, dtype=np.int64)
    cdef cnp.int64_t[:] prim_unsorted = np.zeros(prim.size(), dtype=np.int64)
    cdef cnp.int64_t[:] secd_unsorted = np.zeros(secd.size(), dtype=np.int64)
    
    for i in range(prim.size()):
        j = prim[i]
        prim_unsorted[i] = idx_sorted[j]
        j = secd[i]
        secd_unsorted[i] = idx_sorted[j]
    
    return np.array(prim_unsorted), np.array(secd_unsorted)
