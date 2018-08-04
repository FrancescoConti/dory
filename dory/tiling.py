#
# tiling.py
# Francesco Conti <francesco.conti88@gmail.com>
#
# Copyright (C) 2018 Francesco Conti
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
#

import torch.nn as nn
from __future__ import print_function
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import solver_parameters_pb2

def get_tiling(
    module,
    x_shape,
    buffer_size,
    **kwargs
):
    if type(module) is torch.nn.modules.conv.Conv2d:
        return __get_tiling_conv2d(module, x_shape, buffer_size, **kwargs)
    elif type(module) is torch.nn.modules.conv.Linear:
        return __get_tiling_linear(module, buffer_size, **kwargs)
    elif type(module) is torch.nn.modules.batchnorm.BatchNorm1d or type(module) is torch.nn.modules.batchnorm.BatchNorm2d:
        print("INFO: ignoring batch normalization layer -- assumed to be merged with the previous Conv layer.")
        return None
    elif type(module) is torch.nn.modules.pooling.AvgPool2d or type(module) is torch.nn.modules.pooling.MaxPool2d:
        print("ERROR: todo.")
        return None
    else:
        print("ERROR: %s is not supported yet" % (type(module)))
        return None

def __get_tiling_conv2d(
    module,
    x_shape,
    buffer_size,
    cost_dim=10000,
    cost_w=100,
    cost_n=10,
    cost_h=1,
    max_tile_n_in=None,
    max_tile_n_out=None,
    min_tile_w_in=None,
    min_tile_h_in=None,
    min_tile_w_out=None,
    min_tile_h_out=None,
    ds_x=2,
    ds_y=2,
    ds_W=2
):
    parameters = pywrapcp.Solver.DefaultSolverParameters()
    solver = pywrapcp.Solver("simple_CP", parameters)

    fs = module.kernel_size[0]
    s  = module.stride[0]
    p  = module.padding[0]
    n_in  = module.in_channels
    n_out = module.out_channels
    h_in  = x_shape[0]
    w_in  = x_shape[1]
    h_out = (h_in +2*p -fs+1) // s
    w_out = (w_in +2*p -fs+1) // s

    if max_tile_n_out is None:
        max_tile_n_out = n_out
    if max_tile_n_in is None:
        max_tile_n_in = n_in
    if min_tile_w_in is None:
        min_tile_w_in = 1
    if min_tile_h_in is None:
        min_tile_h_in = 1
    if min_tile_w_out is None:
        min_tile_w_out = 1
    if min_tile_h_out is None:
        min_tile_h_out = 1

    # this is to renormalize all costs
    max_obj_value = buffer_size * cost_dim * 2

    # integer positive variables.
    tile_n_in  = solver.IntVar(1, max_tile_n_in , 'tile_n_in' )
    tile_n_out = solver.IntVar(1, max_tile_n_out, 'tile_n_out')
    tile_h_in  = solver.IntVar(min_tile_h_in, h_in , 'tile_h_in' )
    tile_h_out = solver.IntVar(min_tile_h_out, h_out, 'tile_h_out')
    tile_w_in  = solver.IntVar(min_tile_w_in, w_in , 'tile_w_in' )
    tile_w_out = solver.IntVar(min_tile_w_out, w_out, 'tile_w_out')

    # constraints
    solver.Add(ds_x*tile_n_in*tile_h_in*tile_w_in + ds_y*tile_n_out*tile_h_out*tile_w_out + ds_W*tile_n_in*tile_n_out*fs*fs <= buffer_size)
    solver.Add((tile_h_out + (fs-1)/2 - p) * s == tile_h_in)
    solver.Add((tile_w_out + (fs-1)/2 - p) * s == tile_w_in)

    # objective
    obj_expr = solver.IntVar(0, max_obj_value, "obj_expr")
    solver.Add(obj_expr == cost_dim * (ds_x*tile_n_in*tile_h_in*tile_w_in + ds_y*tile_n_out*tile_h_out*tile_w_out + ds_W*tile_n_in*tile_n_out*fs*fs) 
                         + cost_w   * tile_w_in
                         + cost_h   * tile_h_in
                         + cost_n   * tile_n_in )
    objective = solver.Maximize(obj_expr, 1)

    decision_builder = solver.Phase([tile_n_in, tile_n_out, tile_h_in, tile_h_out, tile_w_in, tile_w_out],
                                     solver.CHOOSE_FIRST_UNBOUND,
                                     solver.ASSIGN_MIN_VALUE)

    # Create a solution collector.
    collector = solver.LastSolutionCollector()
    # Add the decision variables.
    collector.Add(tile_n_in)
    collector.Add(tile_n_out)
    collector.Add(tile_h_in)
    collector.Add(tile_h_out)
    collector.Add(tile_w_in)
    collector.Add(tile_w_out)
    # Add the objective.
    collector.AddObjective(obj_expr)

    solver.Solve(decision_builder, [objective, collector])
    if collector.SolutionCount() > 0:
        best_solution = collector.SolutionCount() - 1

        tile_n_in  = collector.Value(best_solution, tile_n_in )
        tile_n_out = collector.Value(best_solution, tile_n_out)
        tile_h_in  = collector.Value(best_solution, tile_h_in )
        tile_h_out = collector.Value(best_solution, tile_h_out)
        tile_w_in  = collector.Value(best_solution, tile_w_in )
        tile_w_out = collector.Value(best_solution, tile_w_out)

        print(module.name)
        print("\ttiles:\t\tx[%dx%dx%d]\ty[%dx%dx%d]\tW[%dx%dx%dx%d]" % (
            tile_n_in, tile_h_in, tile_w_in,
            tile_n_out, tile_h_out, tile_w_out,
            tile_n_out, tile_n_in, fs, fs
        ))
        print("\ttiles:\tx:%dB\t\ty:%dB\t\tW:%dB" % (
            ds_x*tile_n_in*tile_h_in*tile_w_in,
            ds_y*tile_n_out*tile_h_out*tile_w_out,
            ds_W*tile_n_out*tile_n_in*fs*fs
        ))

def __get_tiling_linear(
    module,
    buffer_size,
    cost_dim=10000,
    cost_n=10,
    max_tile_n_in=None,
    max_tile_n_out=None,
    ds_x=2,
    ds_y=2,
    ds_W=2,
    **kwargs
):
    parameters = pywrapcp.Solver.DefaultSolverParameters()
    solver = pywrapcp.Solver("simple_CP", parameters)

    n_in  = module.in_features
    n_out = module.out_features

    if max_tile_n_out is None:
        max_tile_n_out = n_out
    if max_tile_n_in is None:
        max_tile_n_in = n_in

    # this is to renormalize all costs
    max_obj_value = buffer_size * cost_dim * 2

    # integer positive variables.
    tile_n_in  = solver.IntVar(1, max_tile_n_in , 'tile_n_in' )
    tile_n_out = solver.IntVar(1, max_tile_n_out, 'tile_n_out')

    # constraints
    solver.Add(ds_x*tile_n_in + ds_y*tile_n_out + ds_W*tile_n_in*tile_n_out <= buffer_size)

    # objective
    obj_expr = solver.IntVar(0, max_obj_value, "obj_expr")
    solver.Add(obj_expr == cost_dim * (ds_x*tile_n_in + ds_y*tile_n_out + ds_W*tile_n_in*tile_n_out) 
                         + cost_n   * tile_n_in )
    objective = solver.Maximize(obj_expr, 1)

    decision_builder = solver.Phase([tile_n_in, tile_n_out],
                                     solver.CHOOSE_FIRST_UNBOUND,
                                     solver.ASSIGN_MIN_VALUE)

    # Create a solution collector.
    collector = solver.LastSolutionCollector()
    # Add the decision variables.
    collector.Add(tile_n_in)
    collector.Add(tile_n_out)
    # Add the objective.
    collector.AddObjective(obj_expr)

    solver.Solve(decision_builder, [objective, collector])
    if collector.SolutionCount() > 0:
        best_solution = collector.SolutionCount() - 1

        tile_n_in  = collector.Value(best_solution, tile_n_in )
        tile_n_out = collector.Value(best_solution, tile_n_out)

        print(module.name)
        print("\ttiles:\t\tx[%d]\ty[%d]\tW[%dx%d]" % (
            tile_n_in,
            tile_n_out,
            tile_n_out, tile_n_in,
        ))
        print("\ttiles:\tx:%dB\t\ty:%dB\t\tW:%dB" % (
            ds_x*tile_n_in,
            ds_y*tile_n_out,
            ds_W*tile_n_out*tile_n_in
        ))