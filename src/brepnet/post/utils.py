"""Compatibility exports for post-processing helpers.

`utils.py` used to contain every post-processing helper. The implementation has
been split into focused modules, but this file intentionally keeps the old import
surface alive for existing scripts that still do `from src.brepnet.post.utils import *`.
"""

import random
from itertools import combinations
from random import randint

import numpy as np
import torch
import torch.nn as nn
import trimesh
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Sewing,
)
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.GeomAbs import GeomAbs_C0, GeomAbs_C1, GeomAbs_C2, GeomAbs_C3, GeomAbs_G1
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline, GeomAPI_PointsToBSplineSurface
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.ShapeAnalysis import (
    ShapeAnalysis_FreeBounds,
    ShapeAnalysis_ShapeTolerance,
    ShapeAnalysis_Shell,
    ShapeAnalysis_Wire,
)
from OCC.Core.ShapeExtend import ShapeExtend_WireData
from OCC.Core.ShapeFix import (
    ShapeFix_ComposeShell,
    ShapeFix_Edge,
    ShapeFix_Face,
    ShapeFix_ShapeTolerance,
    ShapeFix_Shell,
    ShapeFix_Solid,
    ShapeFix_Wire,
)
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.Core.TopAbs import (
    TopAbs_COMPOUND,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopAbs_REVERSED,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopTools import TopTools_HSequenceOfShape
from OCC.Core.TopoDS import TopoDS_Builder, TopoDS_Shell, TopoDS_Vertex, topods
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_XYZ
from OCC.Display.SimpleGui import init_display
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from shared.occ_utils import get_primitives
from src.brepnet.post.constants import *
from src.brepnet.post.debug import *
from src.brepnet.post.distance import *
from src.brepnet.post.geometry_optimization import *
from src.brepnet.post.mesh import *
from src.brepnet.post.occ_fit import *
from src.brepnet.post.occ_topology import *
from src.brepnet.post.shape import *
