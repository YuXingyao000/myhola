"""连续边面交替拟合：固定样条基函数，交替更新 UV 与控制点。

阅读顺序：refine → update_uv_correspondence → build_geometry_system → solve_qp。
依赖：numpy、scipy、osqp。全部计算 float64，不导入 Img2Brep 包。
固定结点/次数/正权重；周期基函数折叠共享唯一 poles；关联由调用方提供。
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from functools import cached_property

import numpy as np
import osqp
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.optimize import least_squares


@dataclass(frozen=True)
class Axis:
    knots: np.ndarray
    degree: int
    periodic_count: int | None = None

    @property
    def count(self):
        return self.periodic_count or (len(self.knots) - self.degree - 1)

    @property
    def domain(self):
        return np.array([self.knots[self.degree], self.knots[-self.degree - 1]])

    @cached_property
    def evaluator(self):
        expanded_count = len(self.knots) - self.degree - 1
        coefficients = np.zeros((expanded_count, self.count))
        coefficients[np.arange(expanded_count), np.arange(expanded_count) % self.count] = 1.0
        return BSpline(self.knots, coefficients, self.degree, extrapolate=False)

    def basis(self, parameters, derivative=0):
        if self.periodic_count is not None:
            start, end = self.domain
            parameters = start + np.mod(parameters - start, end - start)
        return self.evaluator(parameters, nu=derivative)


@dataclass
class Spline:
    axes: tuple[Axis, ...]       # 曲线一个轴；曲面两个轴，展平顺序为 u-major。
    poles: np.ndarray            # [control_count, 3]
    weights: np.ndarray          # [control_count]，固定正有理权重

    def _tensor_basis(self, parameters, derivative_axis=None):
        rows = np.ones((len(parameters), 1))
        for axis_id, axis in enumerate(self.axes):
            basis = axis.basis(parameters[:, axis_id], int(axis_id == derivative_axis))
            rows = np.einsum("ni,nj->nij", rows, basis).reshape(len(parameters), -1)
        return rows

    def basis(self, parameters, derivative_axis=None):
        """固定权重后，位置 = rational_basis @ poles；导数用商法则。"""
        weighted = self._tensor_basis(parameters) * self.weights
        denominator = weighted.sum(axis=1, keepdims=True)
        if derivative_axis is None:
            return weighted / denominator
        derivative = self._tensor_basis(parameters, derivative_axis) * self.weights
        return (derivative * denominator - weighted * derivative.sum(axis=1, keepdims=True)) / denominator**2

    def evaluate(self, parameters):
        return self.basis(parameters) @ self.poles

    def control_difference(self):
        """控制网二阶差分；这是改变量正则，不是连续曲率积分。"""
        if len(self.axes) == 1:
            return np.diff(np.eye(self.axes[0].count), n=2, axis=0)
        nu, nv = (axis.count for axis in self.axes)
        return np.vstack((np.kron(np.diff(np.eye(nu), n=2, axis=0), np.eye(nv)),
                          np.kron(np.eye(nu), np.diff(np.eye(nv), n=2, axis=0))))


@dataclass
class Geometry:
    curves: list[Spline]
    surfaces: list[Spline]
    vertices: np.ndarray         # [V, 3]；不是从 GT 读取的顶点


@dataclass(frozen=True)
class Observation:
    parameters: np.ndarray      # 曲线 [N,1]，曲面 [N,2]；固定初始参数
    xyz: np.ndarray             # [N,3]，固定原预测


@dataclass(frozen=True)
class Config:
    iterations: int = 8
    freeze_surface_iterations: int = 1
    boundary_samples: int = 32
    curve_data: float = 1.0
    surface_data: float = 5.0
    vertex_data: float = 1.0
    incidence: float = 100.0
    control_smoothness: float = 0.01
    control_anchor: float = 1e-5
    uv_smoothness: float = 1e-4
    endpoint_alignment: float = 100.0
    curve_move_limit: float = 0.08     # 相对初始 poles 的每坐标盒约束，XYZ 输入单位
    surface_move_limit: float = 0.01
    vertex_move_limit: float = 0.03
    uv_step_fraction: float = 0.15    # 每次 UV 步长 / 对应参数域长度
    constraint_tolerance: float = 2e-7
    descent_tolerance: float = 1e-10


@dataclass(frozen=True)
class Problem:
    reference: Geometry
    curve_observations: list[Observation]
    surface_observations: list[Observation]
    edge_faces: np.ndarray       # [E,2]，每条曲线的两个所属面
    endpoint_vertices: np.ndarray  # [E,2]，起/终点 vertex ID，-1 表示不施加
    uv_reference: np.ndarray     # [E,2,M,2]，固定初始 UV 轨迹
    endpoint_pairs: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=int))


@dataclass(frozen=True)
class Layout:
    curves: list[np.ndarray]
    surfaces: list[np.ndarray]
    vertices: np.ndarray
    size: int


def make_layout(geometry):
    cursor, groups = 0, []
    for splines in (geometry.curves, geometry.surfaces):
        indices = []
        for spline in splines:
            indices.append(np.arange(cursor, cursor + len(spline.poles)))
            cursor += len(spline.poles)
        groups.append(indices)
    vertices = np.arange(cursor, cursor + len(geometry.vertices))
    return Layout(*groups, vertices, cursor + len(vertices))


def pack_geometry(geometry):
    return np.vstack([s.poles for s in geometry.curves + geometry.surfaces] + [geometry.vertices])


def unpack_geometry(geometry, values, layout):
    result = copy.deepcopy(geometry)
    for spline, ids in zip(result.curves + result.surfaces, layout.curves + layout.surfaces):
        spline.poles = values[ids].copy()
    result.vertices = values[layout.vertices].copy()
    return result


def boundary_parameters(curve, config):
    return np.linspace(*curve.axes[0].domain, config.boundary_samples)[:, None]


def embed(rows, columns, size):
    """局部基函数行 → 全体 poles/vertices 的列索引。"""
    local = sparse.coo_matrix(rows)
    return sparse.csr_matrix((local.data, (local.row, columns[local.col])), shape=(local.shape[0], size))


class ResidualSystem:
    """统一存放 sqrt(weight/count) * (A X - b)，所有 block 共用同一目标。"""
    def __init__(self):
        self.rows, self.targets, self.terms = [], [], []

    def add(self, name, rows, targets, weight):
        if rows.shape[0] == 0 or weight == 0:
            return
        scale = np.sqrt(weight / rows.shape[0])
        self.rows.append(rows * scale)
        self.targets.append(np.asarray(targets, dtype=np.float64) * scale)
        self.terms.append(name)

    def matrices(self):
        return sparse.vstack(self.rows, format="csr"), np.vstack(self.targets)

    def values(self, controls):
        values = {}
        for name, rows, targets in zip(self.terms, self.rows, self.targets):
            values[name] = values.get(name, 0.0) + float(np.square(rows @ controls - targets).sum())
        return values


def build_geometry_system(geometry, problem, uv, config):
    """固定 UV：数据、双侧 incidence、控制网改变量，全部成为线性残差。"""
    layout, system = make_layout(geometry), ResidualSystem()
    for name, splines, observations, ids, weight in (
        ("curve_data", geometry.curves, problem.curve_observations, layout.curves, config.curve_data),
        ("surface_data", geometry.surfaces, problem.surface_observations, layout.surfaces, config.surface_data),
    ):
        for spline, observation, columns in zip(splines, observations, ids):
            rows = embed(spline.basis(observation.parameters), columns, layout.size)
            system.add(name, rows, observation.xyz, weight / len(splines))
    for edge, curve in enumerate(geometry.curves):
        curve_rows = embed(curve.basis(boundary_parameters(curve, config)), layout.curves[edge], layout.size)
        for side, face in enumerate(problem.edge_faces[edge]):
            surface_rows = embed(geometry.surfaces[face].basis(uv[edge, side]), layout.surfaces[face], layout.size)
            system.add("incidence", curve_rows - surface_rows, np.zeros((config.boundary_samples, 3)),
                       config.incidence / (2 * len(geometry.curves)))
    endpoint_rows = []
    for edge_a, end_a, edge_b, end_b in problem.endpoint_pairs:
        a, b = geometry.curves[edge_a], geometry.curves[edge_b]
        row_a = embed(a.basis(a.axes[0].domain[end_a:end_a + 1, None]), layout.curves[edge_a], layout.size)
        row_b = embed(b.basis(b.axes[0].domain[end_b:end_b + 1, None]), layout.curves[edge_b], layout.size)
        endpoint_rows.append(row_a - row_b)
    if endpoint_rows:
        system.add("endpoint_alignment", sparse.vstack(endpoint_rows), np.zeros((len(endpoint_rows), 3)),
                   config.endpoint_alignment)
    originals = problem.reference.curves + problem.reference.surfaces
    for spline, original, columns in zip(geometry.curves + geometry.surfaces, originals,
                                         layout.curves + layout.surfaces):
        difference = spline.control_difference()
        system.add("control_smoothness", embed(difference, columns, layout.size),
                   difference @ original.poles, config.control_smoothness / len(originals))
    if len(layout.vertices):
        system.add("vertex_data", embed(np.eye(len(layout.vertices)), layout.vertices, layout.size),
                   problem.reference.vertices, config.vertex_data)
    system.add("control_anchor", sparse.eye(layout.size, format="csr"),
               pack_geometry(problem.reference), config.control_anchor)
    return system


def endpoint_constraints(geometry, problem, layout):
    """B(t_endpoint) Q - V = 0；不假定首尾 pole 一定等于端点。"""
    rows = []
    for edge, curve in enumerate(geometry.curves):
        basis = curve.basis(curve.axes[0].domain[:, None])
        for end, vertex in enumerate(problem.endpoint_vertices[edge]):
            if vertex >= 0:
                rows.append(embed(basis[end:end + 1], layout.curves[edge], layout.size)
                            - embed(np.ones((1, 1)), layout.vertices[vertex:vertex + 1], layout.size))
    return sparse.vstack(rows, format="csr") if rows else sparse.csr_matrix((0, layout.size))


def movement_bounds(problem, config, layout):
    radii = np.zeros(layout.size)
    for columns in layout.curves:
        radii[columns] = config.curve_move_limit
    for columns in layout.surfaces:
        radii[columns] = config.surface_move_limit
    radii[layout.vertices] = config.vertex_move_limit
    original = pack_geometry(problem.reference)
    return original - radii[:, None], original + radii[:, None]


def solve_qp(rows, targets, geometry, problem, config, block):
    """min ||A X-b||²；共享端点等式 + 以初始几何为中心的位移盒。"""
    layout, current = make_layout(geometry), pack_geometry(geometry)
    equality = endpoint_constraints(geometry, problem, layout)
    lower, upper = movement_bounds(problem, config, layout)
    active = np.zeros(layout.size, dtype=bool)
    groups = layout.curves + [layout.vertices] if block == "curves" else layout.surfaces
    for columns in groups:
        active[columns] = True
    lower[~active], upper[~active] = current[~active], current[~active]
    constraints = sparse.vstack((sparse.eye(layout.size), equality), format="csc")
    hessian = sparse.triu(2 * (rows.T @ rows), format="csc")
    linear = -2 * np.asarray(rows.T @ targets)
    values, statuses = current.copy(), []
    for coordinate in range(3):
        solver = osqp.OSQP()
        solver.setup(P=hessian, q=linear[:, coordinate], A=constraints,
                     l=np.r_[lower[:, coordinate], np.zeros(equality.shape[0])],
                     u=np.r_[upper[:, coordinate], np.zeros(equality.shape[0])],
                     eps_abs=1e-9, eps_rel=1e-9, max_iter=20000, polishing=True, verbose=False)
        solver.warm_start(x=current[:, coordinate])
        solution = solver.solve(raise_error=False)
        if solution.info.status_val not in (1, 2) or solution.x is None:
            raise RuntimeError(f"{block} QP: {solution.info.status}; no geometry accepted")
        values[:, coordinate] = solution.x
        statuses.append(solution.info.status)
    equal_error = float(np.max(np.abs(equality @ values), initial=0))
    box_error = float(max(np.max(lower - values, initial=0), np.max(values - upper, initial=0)))
    if not np.isfinite(values).all() or max(equal_error, box_error) > config.constraint_tolerance:
        raise RuntimeError(f"QP constraint residual too large: endpoint={equal_error}, box={box_error}")
    return unpack_geometry(geometry, values, layout), {"qp_status": statuses, "endpoint_error": equal_error,
                                                     "box_violation": box_error}


def initialize_feasible_geometry(geometry, problem, config):
    """Phase I：最小改动接上已指定端点；不把不可行初始状态纳入 best。"""
    layout = make_layout(geometry)
    equality = endpoint_constraints(geometry, problem, layout)
    if np.max(np.abs(equality @ pack_geometry(geometry)), initial=0) <= config.constraint_tolerance:
        return copy.deepcopy(geometry), {"phase_one": "already_feasible"}
    result, diagnostics = solve_qp(sparse.eye(layout.size, format="csr"), pack_geometry(geometry),
                                   geometry, problem, config, "curves")
    return result, {"phase_one": "repaired", **diagnostics}


def uv_regularizer(surface, count):
    spans = np.array([np.ptp(axis.domain) for axis in surface.axes])
    difference = np.diff(np.eye(count), n=2, axis=0)
    return sparse.kron(difference, np.diag(1 / spans), format="csr")


def project_uv_trajectory(surface, xyz, initial_uv, reference_uv, config, edge_count):
    """在连续面上局部投影整条轨迹，固定初始 UV 正则耦合邻近参数点。"""
    count = len(xyz)
    domains = np.array([axis.domain for axis in surface.axes])
    spans = np.diff(domains, axis=1).ravel()
    lower = np.maximum(domains[:, 0], initial_uv - config.uv_step_fraction * spans)
    upper = np.minimum(domains[:, 1], initial_uv + config.uv_step_fraction * spans)
    for axis_id, axis in enumerate(surface.axes):
        if axis.periodic_count is not None:
            lower[:, axis_id] = initial_uv[:, axis_id] - config.uv_step_fraction * spans[axis_id]
            upper[:, axis_id] = initial_uv[:, axis_id] + config.uv_step_fraction * spans[axis_id]
    difference = uv_regularizer(surface, count)
    incidence_scale = np.sqrt(config.incidence / (2 * edge_count * count))
    smooth_scale = np.sqrt(config.uv_smoothness / (2 * edge_count * max(count - 2, 1)))

    def residual(flat_uv):
        geometric = surface.evaluate(flat_uv.reshape(-1, 2)) - xyz
        smooth = difference @ (flat_uv - reference_uv.ravel())
        return np.r_[incidence_scale * geometric.ravel(), smooth_scale * smooth]

    def jacobian(flat_uv):
        uv = flat_uv.reshape(-1, 2)
        derivatives = np.stack([surface.basis(uv, axis) @ surface.poles for axis in (0, 1)], axis=-1)
        row_ids = np.repeat(np.arange(3 * count), 2)
        column_ids = np.broadcast_to(2 * np.arange(count)[:, None, None] + np.arange(2), (count, 3, 2))
        geometric = sparse.coo_matrix((derivatives.ravel(), (row_ids, column_ids.ravel())),
                                      shape=(3 * count, 2 * count))
        return sparse.vstack((incidence_scale * geometric, smooth_scale * difference), format="csr")

    solution = least_squares(residual, initial_uv.ravel(), jac=jacobian,
                             bounds=(lower.ravel(), upper.ravel()), max_nfev=60,
                             ftol=1e-9, xtol=1e-9, gtol=1e-9)
    updated = solution.x.reshape(-1, 2)
    return updated, {"uv_nfev": solution.nfev, "uv_solver_status": solution.status,
                     "uv_step_fraction_max": float(np.max(np.abs(updated - initial_uv) / spans)),
                     "uv_trajectory_increment_max": float(np.linalg.norm(np.diff(updated / spans, axis=0), axis=1).max(initial=0))}


def initialize_uv(geometry, edge_faces, config):
    """粗网格只用于初始化 UV；随后在连续曲面上求值/投影。"""
    result = np.empty((len(geometry.curves), 2, config.boundary_samples, 2))
    for edge, curve in enumerate(geometry.curves):
        xyz = curve.evaluate(boundary_parameters(curve, config))
        for side, face in enumerate(edge_faces[edge]):
            surface = geometry.surfaces[face]
            grid = np.stack(np.meshgrid(*[np.linspace(*axis.domain, 25) for axis in surface.axes],
                                       indexing="ij"), axis=-1).reshape(-1, 2)
            distances = np.square(xyz[:, None] - surface.evaluate(grid)[None]).sum(axis=-1)
            seed = grid[distances.argmin(axis=1)]
            for axis_id, axis in enumerate(surface.axes):
                if axis.periodic_count is not None:
                    seed[:, axis_id] = np.unwrap(seed[:, axis_id], period=np.ptp(axis.domain))
            result[edge, side], _ = project_uv_trajectory(surface, xyz, seed, seed, config, len(geometry.curves))
    return result


def update_uv_correspondence(geometry, problem, uv, config):
    updated, diagnostics = uv.copy(), []
    for edge, curve in enumerate(geometry.curves):
        xyz = curve.evaluate(boundary_parameters(curve, config))
        for side, face in enumerate(problem.edge_faces[edge]):
            updated[edge, side], stats = project_uv_trajectory(
                geometry.surfaces[face], xyz, uv[edge, side], problem.uv_reference[edge, side], config, len(geometry.curves))
            diagnostics.append({"edge": edge, "side": side, **stats})
    return updated, {"uv_updates": diagnostics}


def objective_terms(geometry, problem, uv, config):
    values = build_geometry_system(geometry, problem, uv, config).values(pack_geometry(geometry))
    values["uv_smoothness"] = 0.0
    for edge, owners in enumerate(problem.edge_faces):
        for side, face in enumerate(owners):
            difference = uv_regularizer(geometry.surfaces[face], config.boundary_samples)
            residual = difference @ (uv[edge, side] - problem.uv_reference[edge, side]).ravel()
            values["uv_smoothness"] += config.uv_smoothness * float(residual @ residual) / (
                2 * len(geometry.curves) * max(config.boundary_samples - 2, 1))
    return values


def solve_curve_controls(geometry, problem, uv, config):
    rows, targets = build_geometry_system(geometry, problem, uv, config).matrices()
    return solve_qp(rows, targets, geometry, problem, config, "curves")


def solve_surface_controls(geometry, problem, uv, config):
    rows, targets = build_geometry_system(geometry, problem, uv, config).matrices()
    return solve_qp(rows, targets, geometry, problem, config, "surfaces")


def accept_step(old_geometry, old_uv, new_geometry, new_uv, problem, config, history, iteration, block, stats):
    before = sum(objective_terms(old_geometry, problem, old_uv, config).values())
    terms = objective_terms(new_geometry, problem, new_uv, config)
    after = sum(terms.values())
    accepted = np.isfinite(after) and after <= before + config.descent_tolerance
    history.append({"iteration": iteration, "block": block, "before": before, "candidate": after,
                    "accepted": bool(accepted), "terms": terms, **stats})
    return (new_geometry, new_uv) if accepted else (old_geometry, old_uv)


def refine(geometry, curve_observations, surface_observations, edge_faces, endpoint_vertices,
           config=Config(), *, endpoint_pairs=()):
    """主循环：更新对应 → 固定面调边 → 固定边调面。只有本函数组织顺序。"""
    started = time.perf_counter()
    reference = copy.deepcopy(geometry)
    uv = initialize_uv(reference, edge_faces, config)
    problem = Problem(reference, copy.deepcopy(curve_observations), copy.deepcopy(surface_observations),
                      np.array(edge_faces, dtype=int, copy=True), np.array(endpoint_vertices, dtype=int, copy=True), uv.copy(),
                      np.array(endpoint_pairs, dtype=int, copy=True).reshape(-1, 4))
    raw_initial = measure(reference, problem, uv, config)
    geometry, phase_one = initialize_feasible_geometry(geometry, problem, config)
    initial_feasible = measure(geometry, problem, uv, config)
    best_geometry, best_uv, best_value = copy.deepcopy(geometry), uv.copy(), initial_feasible["objective"]
    history = []
    for iteration in range(config.iterations):
        new_uv, stats = update_uv_correspondence(geometry, problem, uv, config)
        geometry, uv = accept_step(geometry, uv, geometry, new_uv, problem, config, history, iteration, "uv", stats)
        curves_updated, stats = solve_curve_controls(geometry, problem, uv, config)
        geometry, uv = accept_step(geometry, uv, curves_updated, uv, problem, config, history, iteration, "curves", stats)
        if iteration >= config.freeze_surface_iterations:
            surfaces_updated, stats = solve_surface_controls(geometry, problem, uv, config)
            geometry, uv = accept_step(geometry, uv, surfaces_updated, uv, problem, config, history, iteration, "surfaces", stats)
        value = sum(objective_terms(geometry, problem, uv, config).values())
        if value < best_value:
            best_geometry, best_uv, best_value = copy.deepcopy(geometry), uv.copy(), value
    report = {"config": asdict(config), "phase_one": phase_one, "raw_initial": raw_initial,
              "initial_feasible": initial_feasible,
              "final": measure(best_geometry, problem, best_uv, config), "history": history,
              "wall_seconds": time.perf_counter() - started,
              "scope": "continuous spline refinement before OCC trimming; no final STEP validity claim"}
    return best_geometry, best_uv, report, problem


def measure(geometry, problem, uv, config):
    gaps, curve_moves, surface_moves = [], [], []
    for edge, curve in enumerate(geometry.curves):
        domain = curve.axes[0].domain
        t = domain[0] + (np.arange(4 * config.boundary_samples) + 0.5) / (4 * config.boundary_samples) * np.ptp(domain)
        xyz = curve.evaluate(t[:, None])
        curve_moves.extend(np.linalg.norm(xyz - problem.reference.curves[edge].evaluate(t[:, None]), axis=1))
        for side, face in enumerate(problem.edge_faces[edge]):
            interpolated = np.column_stack([np.interp(t, boundary_parameters(curve, config).ravel(), uv[edge, side, :, k]) for k in (0, 1)])
            gaps.extend(np.linalg.norm(xyz - geometry.surfaces[face].evaluate(interpolated), axis=1))
    for surface, original in zip(geometry.surfaces, problem.reference.surfaces):
        grid = np.stack(np.meshgrid(*[np.linspace(*axis.domain, 31) for axis in surface.axes], indexing="ij"), axis=-1).reshape(-1, 2)
        surface_moves.extend(np.linalg.norm(surface.evaluate(grid) - original.evaluate(grid), axis=1))
    layout, controls = make_layout(geometry), pack_geometry(geometry)
    lower, upper = movement_bounds(problem, config, layout)
    terms = objective_terms(geometry, problem, uv, config)
    return {"objective": sum(terms.values()), "terms": terms,
            "dense_correspondence_gap_rms": float(np.sqrt(np.mean(np.square(gaps)))) if gaps else 0.0,
            "dense_correspondence_gap_p95": float(np.percentile(gaps, 95)) if gaps else 0.0,
            "dense_correspondence_gap_max": float(max(gaps, default=0)),
            "curve_displacement_max": float(max(curve_moves, default=0)),
            "surface_displacement_max": float(max(surface_moves, default=0)),
            "endpoint_error": float(np.max(np.abs(endpoint_constraints(geometry, problem, layout) @ controls), initial=0)),
            "box_violation": float(max(np.max(lower - controls, initial=0), np.max(controls - upper, initial=0)))}
