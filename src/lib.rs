use ndarray::{ArrayD, Axis, IxDyn};
use num_traits::{One, Zero};
use std::{
    collections::{BTreeMap, HashMap},
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Index},
};

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

pub struct LinearSystem<T: Clone + Default> {
    pub variables: HashMap<String, ShapedVariable>,
    pub constraints: Vec<EqConstraint<T>>,
    pub total_scalars: usize,
}

impl<T> LinearSystem<T>
where
    T: Clone + AddAssign + From<f32> + Default + Display + One,
{
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            constraints: vec![],
            total_scalars: 0,
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

    pub fn add_eq_constraints<IALE, IAT>(&mut self, exprs: IALE, cs: IAT)
    where
        IALE: Into<ArrayExpr<T>>,
        IAT: Into<ArrayD<T>>,
    {
        let exprs: ArrayExpr<T> = exprs.into();
        let cs: ArrayD<T> = cs.into();
        assert!(exprs.inner.shape() == cs.shape());

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
        assert!(exprs.inner.shape() == cs.shape());

        let name = format!("constraint_{}_slacks", self.constraints.len());
        self.add_var(&name, exprs.inner.shape());
        let slacks = self.index(name);
        self.add_eq_constraints(exprs + slacks.into_arr_expr(), cs);
    }
}

impl<T, IS> Index<IS> for LinearSystem<T>
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
        *self = self.clone() + rhs.clone();
    }
}

#[derive(Clone, Debug)]
pub struct ArrayExpr<T: Default + Display> {
    pub inner: ArrayD<LinearExpression<T>>,
}

impl<T: Display + Default + Clone + AddAssign> ArrayExpr<T> {
    pub fn sum_axis(&self, axis: usize) -> Self {
        let inner = self.inner.sum_axis(Axis(axis));
        Self { inner }
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

impl<T> Add for ArrayExpr<T>
where
    T: Clone + AddAssign + Default + Display,
{
    type Output = ArrayExpr<T>;

    fn add(mut self, rhs: Self) -> Self::Output {
        for (le, re) in self.inner.iter_mut().zip(rhs.inner.iter()) {
            *le += re.clone();
        }
        self
    }
}

pub struct EqConstraint<T: Clone + Default> {
    expr: LinearExpression<T>,
    c: T,
}

impl<T: Clone + Default + Display> Debug for EqConstraint<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = format!("{:?} == {}", self.expr, self.c);
        f.write_str(s.as_str())
    }
}

#[test]
fn example() {
    let mut ls = LinearSystem::<f32>::new();
    ls.add_var("v1", &[2, 2]);
    let v1 = &ls["v1"];

    // prints scalar ids
    dbg!(v1);

    let a = ndarray::array![[1.0, 2.0], [3.0, 4.0]].into_dyn();

    // v1 + a.*v1 matrix expression
    let exprs2 = v1.into_arr_expr() + (a, v1).into();
    dbg!(&exprs2);

    ls.add_leq_constraints(exprs2.sum_axis(0), ndarray::array![7.0, 8.0].into_dyn());
    dbg!(&ls.constraints);
}
