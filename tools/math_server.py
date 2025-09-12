# tools/math_server.py
import sys
import json
import tempfile
import os
import time
import uuid
from typing import Any, Dict, List, Union

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np


# ---------------------- 解析与公用 ----------------------
TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

def sanitize_expr(expr: str) -> str:
    if not expr:
        return expr
    # 兼容用户输入 ^ 作为幂
    expr = expr.replace("^", "**")
    return expr.strip()

def to_sympy_expr(expr: str) -> sp.Expr:
    expr = sanitize_expr(expr)
    # 允许隐式乘法：3x -> 3*x, 2(x+1) -> 2*(x+1)
    return parse_expr(expr, transformations=TRANSFORMS)

def reply(msg_id, result=None, error=None):
    resp = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        resp["error"] = {"code": -32000, "message": str(error)}
    else:
        resp["result"] = result
    print(json.dumps(resp, ensure_ascii=False), flush=True)


# ---------------------- 工具实现 ----------------------
def tool_solve_equation(args: dict):
    """
    支持：
      expr="x^2+3x+2"   -> 视为 =0
      expr="x^2+3x+2=0"
    var: 默认 "x"
    """
    expr = args.get("expr", "")
    var = args.get("var", "x")
    x = sp.symbols(var)

    try:
        expr = sanitize_expr(expr)
        if "=" in expr:
            lhs_s, rhs_s = expr.split("=", 1)
            lhs = to_sympy_expr(lhs_s)
            rhs = to_sympy_expr(rhs_s)
            eq = sp.Eq(lhs, rhs)
            sols = sp.solve(eq, x)
        else:
            e = to_sympy_expr(expr)
            sols = sp.solve(sp.Eq(e, 0), x)
        return {"solutions": [str(s) for s in sols]}
    except Exception as e:
        raise ValueError(f"表达式解析/求解失败: {expr}，错误: {e}")

def tool_diff(args: dict):
    expr = args.get("expr", "")
    var = args.get("var", "x")
    x = sp.symbols(var)
    try:
        deriv = sp.diff(to_sympy_expr(expr), x)
        return {"derivative": str(deriv)}
    except Exception as e:
        raise ValueError(f"求导失败: {expr}，错误: {e}")

def tool_integrate(args: dict):
    expr = args.get("expr", "")
    var = args.get("var", "x")
    x = sp.symbols(var)
    try:
        integ = sp.integrate(to_sympy_expr(expr), x)
        return {"integral": str(integ)}
    except Exception as e:
        raise ValueError(f"积分失败: {expr}，错误: {e}")

# --------- 升级版 matrix_multiply（保留向后兼容字段名） ---------
def _validate_matrix_list(M: Any, name: str):
    if not isinstance(M, list) or not all(isinstance(row, list) for row in M):
        raise ValueError(f"{name} 必须是二维列表（list of lists）。")
    if len(M) == 0 or len(M[0]) == 0:
        raise ValueError(f"{name} 不能为空，且至少包含一列。")
    n = len(M[0])
    if any(len(row) != n for row in M):
        raise ValueError(f"{name} 的每一行长度必须一致。")

def tool_matrix_multiply(args: dict):
    """
    入参:
      A: list[list[number]]
      B: list[list[number]]
    返回(向后兼容):
      {
        "result": list[list[number]],   # 兼容旧字段
        "matrix": list[list[number]],   # 新字段
        "shape": [m, p]
      }
    """
    try:
        A = args.get("A", [])
        B = args.get("B", [])
        _validate_matrix_list(A, "A")
        _validate_matrix_list(B, "B")

        m, nA = len(A), len(A[0])
        nB, p = len(B), len(B[0])
        if nA != nB:
            raise ValueError(f"无法相乘：A 的列数({nA}) 必须等于 B 的行数({nB})。")

        # 尝试用 numpy（更快）；失败则退回 sympy
        C_list: List[List[Union[float, str]]]
        try:
            A_np = np.asarray(A, dtype=float)
            B_np = np.asarray(B, dtype=float)
            C_np = A_np @ B_np
            C_list = C_np.tolist()
        except Exception:
            A_sp = sp.Matrix(A)
            B_sp = sp.Matrix(B)
            C_sp = A_sp * B_sp
            # 首先尝试转 float；若含符号就转字符串
            try:
                C_list = [[float(C_sp[i, j]) for j in range(C_sp.shape[1])]
                          for i in range(C_sp.shape[0])]
            except Exception:
                C_list = [[str(C_sp[i, j]) for j in range(C_sp.shape[1])]
                          for i in range(C_sp.shape[0])]

        return {"result": C_list, "matrix": C_list, "shape": [m, p]}
    except Exception as e:
        raise ValueError(f"矩阵乘法失败，错误: {e}")

# --------- 升级版 plot（保留向后兼容字段名） ---------
def tool_plot(args: dict):
    """
    入参:
      expr: str  表达式，如 "sin(x)/x"
      var:  str  变量名，默认 "x"
      start: float 区间起点
      end:   float 区间终点
      可选:
        num: int=600  采样点
        dpi: int=144
        width: float=6.0  (英寸)
        height: float=4.0 (英寸)
        out_dir: str="data/plots"
    返回(向后兼容):
      {
        "image_path": ".../xxx.png",  # 兼容旧字段
        "path": ".../xxx.png",
        "width_px": int,
        "height_px": int,
        "x_range": [start, end],
        "samples": int
      }
    """
    expr = args.get("expr", "")
    var = args.get("var", "x")
    start = float(args.get("start", -10))
    end = float(args.get("end", 10))
    num = int(args.get("num", 600))
    dpi = int(args.get("dpi", 144))
    width = float(args.get("width", 6.0))
    height = float(args.get("height", 4.0))
    out_dir = args.get("out_dir", "data/plots")

    if start >= end:
        raise ValueError(f"绘图区间不合法：start({start}) 必须小于 end({end})。")
    if num < 50:
        raise ValueError("采样点太少，num 建议 ≥ 50。")

    try:
        x = sp.symbols(var)
        f_expr = to_sympy_expr(expr)

        # numpy 后端数值化
        try:
            f = sp.lambdify(x, f_expr, modules=["numpy"])
        except Exception as e:
            raise ValueError(f"无法数值化该表达式：{e}")

        X = np.linspace(float(start), float(end), num=num)
        with np.errstate(all="ignore"):
            Y = f(X)

        # 将数据整理为可绘
        Y = np.asarray(Y)
        # 复数：若虚部很小则取实部；否则置为 NaN
        if np.iscomplexobj(Y):
            imag_small = np.abs(Y.imag) < 1e-10
            Y = np.where(imag_small, Y.real, np.nan)
        # 非有限值置为 NaN
        Y = np.where(np.isfinite(Y), Y, np.nan)

        # 若全为 NaN，报错
        if not np.isfinite(Y).any():
            raise ValueError("该函数在给定区间内没有可绘制的有限实数值。")

        # 输出路径
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
        out_path = os.path.join(out_dir, fname)

        # 画图
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        try:
            plt.plot(X, Y, linewidth=1.5)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.xlabel(var)
            plt.ylabel(str(f_expr))
            plt.title(f"{f_expr}  on  [{start}, {end}]")
            plt.tight_layout()
            fig.savefig(out_path)
        finally:
            plt.close(fig)

        out_path_norm = out_path.replace("\\", "/")
        return {
            "image_path": out_path_norm,         # 兼容旧字段
            "path": out_path_norm,
            "width_px": int(width * dpi),
            "height_px": int(height * dpi),
            "x_range": [float(start), float(end)],
            "samples": int(num),
        }
    except Exception as e:
        raise ValueError(f"绘图失败: {expr}，错误: {e}")


# ---------------------- JSON-RPC 主循环 ----------------------
def handle_call(name: str, args: dict):
    if name == "math.solve_equation":
        return tool_solve_equation(args)
    elif name == "math.diff":
        return tool_diff(args)
    elif name == "math.integrate":
        return tool_integrate(args)
    elif name == "math.matrix_multiply":
        return tool_matrix_multiply(args)
    elif name == "math.plot":
        return tool_plot(args)
    else:
        raise ValueError(f"未知工具: {name}")

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        mid = req.get("id")
        method = req.get("method")
        params = req.get("params", {}) or {}

        try:
            if method == "initialize":
                reply(mid, {"server": "math-tools", "version": "0.5"})
            elif method == "tools/list":
                reply(mid, {"tools": [
                    {"name": "math.solve_equation", "desc": "解方程（支持隐式乘法与 ^ ）"},
                    {"name": "math.diff", "desc": "求导"},
                    {"name": "math.integrate", "desc": "积分"},
                    {"name": "math.matrix_multiply", "desc": "矩阵乘法"},
                    {"name": "math.plot", "desc": "绘制函数图像"}
                ]})
            elif method == "tools/call":
                name = params.get("name")
                args = params.get("arguments", {}) or {}
                result = handle_call(name, args)
                reply(mid, result)
            else:
                reply(mid, error=f"未知方法: {method}")
        except Exception as e:
            reply(mid, error=str(e))

if __name__ == "__main__":
    main()
