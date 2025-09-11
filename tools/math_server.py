# tools/math_server.py
import sys
import json
import tempfile
import os

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------- 解析与公用 ----------------------
TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

def sanitize_expr(expr: str) -> str:
    if not expr:
        return expr
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

def tool_matrix_multiply(args: dict):
    try:
        A = sp.Matrix(args.get("A", []))
        B = sp.Matrix(args.get("B", []))
        result = A * B
        return {"result": result.tolist()}
    except Exception as e:
        raise ValueError(f"矩阵乘法失败，错误: {e}")

def tool_plot(args: dict):
    import numpy as np
    expr = args.get("expr", "")
    var = args.get("var", "x")
    start = float(args.get("start", -10))
    end = float(args.get("end", 10))

    try:
        x = sp.symbols(var)
        f_expr = to_sympy_expr(expr)
        f = sp.lambdify(x, f_expr, "numpy")

        xs = np.linspace(start, end, 400)
        ys = f(xs)
        ys = np.real_if_close(ys)
        ys = np.where(np.isfinite(ys), ys, np.nan)

        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel(var)
        plt.ylabel(str(f_expr))
        plt.title(f"{f_expr} from {start} to {end}")

        fd, tmpfile = tempfile.mkstemp(suffix=".png", prefix="plot_")
        os.close(fd)
        plt.savefig(tmpfile)
        plt.close()

        return {"image_path": tmpfile}
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
                reply(mid, {"server": "math-tools", "version": "0.4"})
            elif method == "tools/list":
                reply(mid, {"tools": [
                    {"name": "math.solve_equation", "desc": "解方程（支持隐式乘法与^）"},
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
