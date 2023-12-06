use std::collections::HashMap;

use minilp::{ComparisonOp, OptimizationDirection, Problem};
use ndarray::{Array1, Array2};

pub struct StandardFormLP<T> {
    /// gradient/cost coeffs
    pub c: Array1<T>,
    /// constraint coeffs
    pub a: Array2<T>,
    /// constraint limits
    pub b: Array1<T>,
}

pub struct LPOutput<T> {
    pub sol: Array1<T>,
    pub solver_stats: HashMap<String, T>,
}

pub trait SolvesLP<T> {
    /// soves min  c^T x  s.t.  a x = b (x>=0)
    fn solve(lp: StandardFormLP<T>) -> Result<LPOutput<T>, String>;
}

pub struct MiniLPSolver;

impl SolvesLP<f64> for MiniLPSolver {
    fn solve(StandardFormLP { c, a, b }: StandardFormLP<f64>) -> Result<LPOutput<f64>, String> {
        let n_vars = c.len();
        let n_constraints = b.len();
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        // equivalent (usize) representation but has private fields
        let minilp_vars: Vec<_> = (0..n_vars)
            .map(|i| problem.add_var(c[i], (0.0, f64::INFINITY)))
            .collect();
        for j in 0..n_constraints {
            let minilp_cons: Vec<_> = (0..n_vars).map(|i| (minilp_vars[i], a[(j, i)])).collect();
            problem.add_constraint(minilp_cons.as_slice(), ComparisonOp::Eq, b[j]);
        }
        match problem.solve() {
            Ok(s) => {
                let sol = minilp_vars.iter().map(|v| s[*v]).collect();
                // TODO: pass some stats through...
                let solver_stats = HashMap::new();
                Ok(LPOutput { sol, solver_stats })
            }
            Err(e) => Err(e.to_string()),
        }
    }
}
