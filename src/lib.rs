use lp_wrap::{LPOutput, SolvesLP, StandardFormLP};
use ndarray::{Array1, Array2, ArrayD, Axis, IxDyn};
use num_traits::{Float, One, Zero};
use std::{
    collections::{BTreeMap, HashMap},
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Index, MulAssign},
};

mod lp_wrap;

/// holds array of scalar variable ids
#[derive(Clone, Debug)]
pub struct ShapedVariable {
    pub inner: ArrayD<usize>,
}

impl ShapedVariable {
    pub fn into_arr_expr<T>(&self) -> ArrayExpr<T>
    where
        T: Clone + AddAssign + From<f32> + Default + Display + One,
    {
        self.clone().into()
    }

    pub fn shaped_sol<T: Clone + Zero>(&self, sol: &Array1<T>) -> ArrayD<T> {
        let mut vals = ArrayD::<T>::zeros(self.inner.shape());
        for (idx, i) in self.inner.indexed_iter() {
            vals[idx] = sol[*i].clone();
        }
        vals
    }
}

impl<T> Into<ArrayExpr<T>> for ShapedVariable
where
    T: Clone + AddAssign + From<f32> + Default + Display + One,
{
    fn into(self) -> ArrayExpr<T> {
        let coefficients = ArrayD::<T>::ones(self.inner.shape());
        (coefficients, &self).into()
    }
}

pub struct LinearProgram<T: Clone + Default> {
    pub variables: HashMap<String, ShapedVariable>,
    pub cost: LinearExpression<T>,
    pub constraints: Vec<EqConstraint<T>>,
    pub total_scalars: usize,
}

impl<T> LinearProgram<T>
where
    T: Clone + AddAssign + MulAssign + From<f32> + Default + Display + Float,
{
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            constraints: vec![],
            total_scalars: 0,
            cost: LinearExpression::default(),
        }
    }

    pub fn add_var<IS>(&mut self, name: IS, shape: &[usize])
    where
        IS: Into<String>,
    {
        let name = name.into();
        assert!(!self.variables.contains_key(&name));

        let dim_prod = shape.iter().fold(1, |acc, n| acc * *n);
        let mut inner = ArrayD::zeros(IxDyn(&shape));
        for (i, v) in inner.iter_mut().enumerate() {
            *v = self.total_scalars + i;
        }
        self.variables.insert(name, ShapedVariable { inner });
        self.total_scalars += dim_prod;
    }

    pub fn set_cost(&mut self, cost: LinearExpression<T>) {
        self.cost = cost;
    }

    pub fn add_eq_constraints<IALE, IAT>(&mut self, exprs: IALE, cs: IAT)
    where
        IALE: Into<ArrayExpr<T>>,
        IAT: Into<ArrayD<T>>,
    {
        let exprs: ArrayExpr<T> = exprs.into();
        let cs: ArrayD<T> = cs.into();
        if !(exprs.inner.len() == 1 && cs.len() == 1) {
            assert!(exprs.inner.shape() == cs.shape());
        }

        for (expr, c) in exprs.inner.into_iter().zip(cs.into_iter()) {
            self.constraints.push(EqConstraint { expr, c })
        }
    }

    pub fn add_leq_constraints<IALE, IAT>(&mut self, exprs: IALE, cs: IAT)
    where
        IALE: Into<ArrayExpr<T>>,
        IAT: Into<ArrayD<T>>,
    {
        let exprs: ArrayExpr<T> = exprs.into();
        let cs: ArrayD<T> = cs.into();
        if !(exprs.inner.len() == 1 && cs.len() == 1) {
            assert!(exprs.inner.shape() == cs.shape());
        }

        let name = format!("constraint_{}_slacks", self.constraints.len());
        self.add_var(&name, exprs.inner.shape());
        let slacks = self.index(name);
        self.add_eq_constraints(exprs + slacks.into_arr_expr(), cs);
    }

    pub fn add_eq_0_constraints<IALE>(&mut self, exprs: IALE)
    where
        IALE: Into<ArrayExpr<T>>,
    {
        let exprs: ArrayExpr<T> = exprs.into();
        let cs = ArrayD::<T>::zeros(exprs.inner.shape());
        self.add_eq_constraints(exprs, cs);
    }

    pub fn add_leq_0_constraints<IALE>(&mut self, exprs: IALE)
    where
        IALE: Into<ArrayExpr<T>>,
    {
        let exprs: ArrayExpr<T> = exprs.into();
        let cs = ArrayD::<T>::zeros(exprs.inner.shape());
        self.add_leq_constraints(exprs, cs);
    }

    pub fn to_standard_form(&self) -> StandardFormLP<T> {
        let c = (0..self.total_scalars)
            .map(|i| self.cost.get_coef(i))
            .collect();
        let n_constraints = self.constraints.len();
        let mut a = Array2::<T>::zeros([n_constraints, self.total_scalars]);
        let mut b = Array1::<T>::zeros(n_constraints);
        for (j, EqConstraint { expr, c }) in self.constraints.iter().enumerate() {
            for i in 0..self.total_scalars {
                a[(j, i)] = expr.get_coef(i);
            }
            b[j] = *c;
        }
        StandardFormLP { c, a, b }
    }

    pub fn solve<Solver: SolvesLP<T>>(&self) -> Result<LPOutput<T>, String> {
        Solver::solve(self.to_standard_form())
    }
}

impl<T, IS> Index<IS> for LinearProgram<T>
where
    T: Clone + Default,
    IS: Into<String>,
{
    type Output = ShapedVariable;

    fn index(&self, index: IS) -> &Self::Output {
        &self.variables[&index.into()]
    }
}

#[derive(Clone, Default)]
pub struct LinearExpression<T: Default> {
    coefficients: BTreeMap<usize, T>,
}

impl<T> LinearExpression<T>
where
    T: Default + Display + Zero + Clone + Copy + MulAssign,
{
    pub fn get_coef(&self, idx: usize) -> T {
        match self.coefficients.get(&idx) {
            Some(c) => c.clone(),
            None => T::zero(),
        }
    }

    pub fn scale(&mut self, s: T) {
        for v in self.coefficients.values_mut() {
            *v *= s;
        }
    }

    pub fn scaled(&self, s: T) -> Self {
        let mut c = self.clone();
        c.scale(s);
        c
    }

    pub fn repeat(&self, shape: &[usize]) -> ArrayExpr<T> {
        let mut inner = ArrayD::default(IxDyn(shape));
        inner.fill(self.clone());
        ArrayExpr { inner }
    }
}

impl<T: Default + Clone + AddAssign> Zero for LinearExpression<T> {
    fn zero() -> Self {
        Self::default()
    }

    fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }
}

impl<T: Default + Display> Debug for LinearExpression<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = String::new();
        for (id, c) in self.coefficients.iter() {
            if out.is_empty() {
                out += format!("{c}s_{id}").as_str();
            } else {
                out += format!(" + {c}s_{id}").as_str();
            }
        }
        f.write_str(out.as_str())
    }
}

impl<T: Clone + AddAssign + Default> Add for LinearExpression<T> {
    type Output = LinearExpression<T>;

    fn add(mut self, rhs: Self) -> Self::Output {
        for (id, rcoef) in rhs.coefficients {
            match self.coefficients.get_mut(&id) {
                Some(lcoef) => *lcoef += rcoef,
                None => {
                    self.coefficients.insert(id, rcoef);
                }
            }
        }
        self
    }
}

impl<T: Clone + AddAssign + Default> AddAssign for LinearExpression<T> {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

#[derive(Clone, Debug, Default)]
pub struct ArrayExpr<T: Default + Display> {
    pub inner: ArrayD<LinearExpression<T>>,
}

impl<T> ArrayExpr<T>
where
    T: Display + Default + Clone + Copy + MulAssign + Add + AddAssign + Zero,
{
    pub fn new(size: &[usize]) -> Self {
        ArrayExpr {
            inner: ArrayD::default(IxDyn(size)),
        }
    }

    pub fn sum_axis(&self, axis: usize) -> Self {
        let inner = self.inner.sum_axis(Axis(axis));
        Self { inner }
    }

    pub fn sum(&self) -> LinearExpression<T> {
        let mut res = self.clone();
        while res.inner.shape().len() > 0 {
            res = res.sum_axis(0);
        }
        res.inner.first().unwrap().clone()
    }

    pub fn scale(&mut self, s: T) {
        for v in self.inner.iter_mut() {
            v.scale(s);
        }
    }

    pub fn scaled(&self, s: T) -> Self {
        let mut c = self.clone();
        c.scale(s);
        c
    }
}

// To get around multiplication of different types
impl<T> From<(ArrayD<T>, &ShapedVariable)> for ArrayExpr<T>
where
    T: Clone + AddAssign + Default + Display,
{
    fn from((coefs, var): (ArrayD<T>, &ShapedVariable)) -> Self {
        assert!(coefs.shape() == var.inner.shape());
        let mut inner = ArrayD::default(IxDyn(&var.inner.shape()));
        for (c, (idx, id)) in coefs.iter().zip(var.inner.indexed_iter()) {
            let mut coefficients = BTreeMap::new();
            coefficients.insert(*id, c.clone());
            inner[idx] = LinearExpression::<T> { coefficients };
        }
        ArrayExpr { inner }
    }
}

impl<T> From<(ArrayD<T>, &ArrayExpr<T>)> for ArrayExpr<T>
where
    T: Clone + Copy + AddAssign + Default + Display + Zero + MulAssign,
{
    fn from((coefs, exprs): (ArrayD<T>, &ArrayExpr<T>)) -> Self {
        assert!(coefs.shape() == exprs.inner.shape());
        let mut inner = ArrayD::default(IxDyn(&exprs.inner.shape()));
        for (c, (idx, expr)) in coefs.iter().zip(exprs.inner.indexed_iter()) {
            inner[idx] = expr.scaled(*c);
        }
        ArrayExpr { inner }
    }
}

impl<T> Add for ArrayExpr<T>
where
    T: Clone + AddAssign + Default + Display,
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        for (le, re) in self.inner.iter_mut().zip(rhs.inner.iter()) {
            *le += re.clone();
        }
        self
    }
}

impl<T> AddAssign for ArrayExpr<T>
where
    T: Clone + AddAssign + Default + Display,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

pub struct EqConstraint<T: Clone + Default> {
    pub expr: LinearExpression<T>,
    pub c: T,
}

impl<T: Clone + Default + Display> Debug for EqConstraint<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = format!("{:?} == {}", self.expr, self.c);
        f.write_str(s.as_str())
    }
}

#[test]
fn toy_prob() {
    let mut ls = LinearProgram::<f64>::new();
    ls.add_var("X", &[2, 2]);
    let x = ls["X"].clone();
    let x_expr = x.into_arr_expr::<f64>();

    let cs = ndarray::array![[0.28029004, 0.4412583], [0.05902943, 0.21942223]].into_dyn();
    let bs = ndarray::array![0.2320803, 0.18310474].into_dyn();
    let hs = ndarray::array![0.53526838, 0.82200931].into_dyn();

    let cs_mul_x: ArrayExpr<f64> = (cs, &x).into();
    ls.set_cost(cs_mul_x.sum());
    dbg!(&ls.cost);
    ls.add_eq_constraints(x_expr.sum_axis(0), bs);
    ls.add_leq_constraints(x_expr.sum_axis(1), hs);
    dbg!(&ls.constraints);

    let LPOutput { sol, solver_stats } = ls.solve::<lp_wrap::MiniLPSolver>().unwrap();
    dbg!(&sol, &solver_stats);

    dbg!(x.shaped_sol(&sol));
}
