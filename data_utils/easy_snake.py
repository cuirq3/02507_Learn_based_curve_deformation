#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 7 20:21:26 2019

@author: vand
"""

import numpy as np
import scipy.interpolate
import scipy.linalg
import sklearn.cluster
import skimage.measure


def initialize_snake(N, center, radius):
    """Initialize circular snake."""
    center = np.array(center).reshape([2, 1])
    angles = np.arange(N) * 2 * np.pi / N
    return center+radius*np.stack((np.cos(angles), np.sin(angles)))


def normalize(n):
    return n/np.sqrt((n**2).sum(axis=0))


def snake_normals(snake):
    """Returns snake normals."""
    ds = normalize(np.roll(snake, 1, axis=1) - snake) 
    tangent = normalize(np.roll(ds, -1, axis=1) + ds)
    normal = tangent[[1,0],:] * np.array([-1,1]).reshape([2,1])
    return normal


def distribute_points(snake, N=None):
    """Distributes snake points equidistantly."""
    if N is None:
        N = snake.shape[1]
    d = np.sqrt(((np.roll(snake, -1, axis=1) - snake)**2).sum(axis=0)) # length of line segments
    f = scipy.interpolate.interp1d(np.r_[0, np.cumsum(d)], 
                                   np.c_[snake, snake[:,0:1]])
    return f(sum(d)*np.arange(N)/N)


def is_crossing(p1, p2, p3, p4):
    """Check if the line segments (p1, p2) and (p3, p4) cross."""
    crossing = False
    d21 = p2 - p1
    d43 = p4 - p3
    d31 = p3 - p1
    det = d21[0]*d43[1] - d21[1]*d43[0] # Determinant
    if det != 0.0 and d21[0] != 0.0 and d21[1] != 0.0:
        a = d43[0]/d21[0] - d43[1]/d21[1]
        b = d31[1]/d21[1] - d31[0]/d21[0]
        if a != 0.0:
            u = b/a
            if d21[0] > 0:
                t = (d43[0]*u + d31[0])/d21[0]
            else:
                t = (d43[1]*u + d31[1])/d21[1]
            crossing = 0 < u < 1 and 0 < t < 1         
    return crossing


def is_counterclockwise(snake):
    """Check if points are ordered counterclockwise."""
    return np.dot(snake[0,1:] - snake[0,:-1],
                  snake[1,1:] + snake[1,:-1]) < 0


def remove_intersections(snake):
    """Reorder snake points to remove self-intersections."""
    pad_snake = np.append(snake, snake[:,0].reshape(2,1), axis=1)
    pad_n = pad_snake.shape[1]
    n = pad_n - 1 
    
    for i in range(pad_n - 3):
        for j in range(i + 2, pad_n - 1):
            pts = pad_snake[:,[i, i + 1, j, j + 1]]
            if is_crossing(pts[:,0], pts[:,1], pts[:,2], pts[:,3]):
                # Reverse vertices of smallest loop
                rb = i + 1 # Reverse begin
                re = j     # Reverse end
                if j - i > n // 2:
                    # Other loop is smallest
                    rb = j + 1
                    re = i + n                    
                while rb < re:
                    ia = rb % n
                    rb = rb + 1                    
                    ib = re % n
                    re = re - 1                    
                    pad_snake[:,[ia, ib]] = pad_snake[:,[ib, ia]]                    
                pad_snake[:,-1] = pad_snake[:,0]                
    snake = pad_snake[:,:-1]
    if is_counterclockwise(snake):
        return snake
    else:
        return np.flip(snake, axis=1)
    
    
def keep_snake_inside(snake, shape):
    """Contains snake insite the image."""
    snake[snake<0]=0
    snake[0][snake[0]>shape[0]-1] = shape[0]-1 
    snake[1][snake[1]>shape[1]-1] = shape[1]-1 
    return snake

    
def regularization_matrix(N, alpha, beta):
    """Matrix for smoothing the snake."""
    d = alpha*np.array([-2, 1, 0, 0]) + beta*np.array([-6, 4, -1, 0])
    D = np.fromfunction(lambda i, j: np.minimum((i-j)%N,(j-i)%N), (N,N), dtype=np.int)
    A = d[np.minimum(D, len(d) - 1)]
    return(scipy.linalg.inv(np.eye(N) - A))


def pixel_clustering(rgb, K):
    """K-means clustering of rgb values."""
    kmeans = sklearn.cluster.KMeans(n_clusters=K)
    i,j,l = rgb.shape
    rgb = rgb.reshape((-1, l))
    kmeans.fit(rgb)
    clusters = kmeans.labels_
    clusters = clusters.reshape((i, j))
    return clusters


def clusters_2_probabilities(clusters, K, mask):
    """Pixel assignment to inside-outside probability."""
    edges = np.arange(K+1)-0.5
    h_in = np.histogram(clusters[mask], bins=edges)[0]/np.sum(mask)
    h_out = np.histogram(clusters[~mask], bins=edges)[0]/np.sum(~mask)
    h_sum = h_in + h_out
    h_sum[h_sum==0] = 1 # should not occur
    p_in = h_in/h_sum
    P_in = p_in[clusters]
    return P_in

def largest_contour(mask):
    '''Largest contour given by binary mask'''
    mask = np.pad(mask, ((1, 1),)) # to make sure that contour is closed
    contours = skimage.measure.find_contours(mask, 0.5)
    l = np.array([c.shape[0] for c in contours])
    i = np.argmax(l)
    contour = contours[i].T - 1 # remove 1 because of padding
    return contour