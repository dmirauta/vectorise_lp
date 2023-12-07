# Vectorise Linear Program

Generate the 1-D representation for LPs with n-D variables (such as the Optimal Transport problem).

`lib.rs` TLDR:

- An n-D variable is represented as an ndarray of scalar ids.
- A linear expression is a collection of coefficients (corresponding to ids).
- The variable `[s_0 s_1]` transforms to the array of expressions `[1s_0, 1s_1]`.
- An elementwise product can be performed between a coefficient ndarray and a variable with `([1.5, 3.2], [s_0, s_1]).into()` producing `[1.5s_0, 3.2s_1]`.
- Expression can be added and Array Expression can be summed along axes.
- Slack variables will be added to inequality constraints to convert them to equalities.
- problem is converted to standard form and passed to a solver, currently only [minilp](https://crates.io/crates/minilp)
- solution is reshaped into original format

# Example

```rust
let mut ls = LinearProgram::<f32>::new();
ls.add_var("v1", &[2, 2]);
let v1 = &ls["v1"];

let a = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();

// v1 + a.*v1 matrix expression
let exprs = v1.into_arr_expr() + (a, v1).into();

ls.add_leq_constraints(exprs.sum_axis(0), array![7.0, 8.0].into_dyn());
dbg!(&ls.constraints);
```

Producing:

```
&ls.constraints = [
    2s_0 + 4s_2 + 1s_4 == 7,
    3s_1 + 5s_3 + 1s_5 == 8,
]
```

See also `toy_problem` in `lib.rs`.

# TODO

- other backends (behind feature flags?)
- prune variables and constraints
- add geq constraints method for convenience
