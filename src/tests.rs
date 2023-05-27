#[cfg(test)]
use crate::Matrix;

#[test]
fn mul_test() {
    let a = Matrix::from(vec![vec![1, 2, 4], vec![3, 4, 9]]).unwrap();
    let b = Matrix::from(vec![vec![1, 2], vec![2, 3], vec![5, 1]]).unwrap();
    let mut c = Matrix::from(vec![vec![25, 12], vec![56, 27]]).unwrap();
    let d = Matrix::from(vec![vec![75, 36], vec![168, 81]]).unwrap();

    assert_eq!(a * b, c);

    c.mul_scalar(3);
    assert_eq!(c, d);
}

#[test]
fn add_sub_test() {
    let a = Matrix::from(vec![vec![1, 2, 3], vec![0, 1, 2]]).unwrap();
    let b = Matrix::from(vec![vec![0, 0, 1], vec![2, 1, 3]]).unwrap();
    let c = Matrix::from(vec![vec![1, 2, 4], vec![2, 2, 5]]).unwrap();
    let d = Matrix::from(vec![vec![1, 2, 2], vec![-2, 0, -1]]).unwrap();
    let e = Matrix::from(vec![vec![-1, -2, -4], vec![-2, -2, -5]]).unwrap();

    assert_eq!(a.clone() + b.clone(), c);
    assert_eq!(a - b, d);
    assert_eq!(-c, e);
}

#[test]
fn det_trace_test() {
    let a = Matrix::from(vec![vec![1, 2, 0], vec![0, 3, 5], vec![0, 0, 10]]).unwrap();
    let b = Matrix::from(vec![vec![1, 2, 0], vec![0, 3, 5]]).unwrap();
    let c = Matrix::from(vec![
        vec![0.0, 0.0, 10.0],
        vec![0.0, 3.0, 5.0],
        vec![1.0, 2.0, 0.0],
    ])
    .unwrap();

    assert_eq!(a.det(), Ok(30));
    assert_eq!(c.det_in_field(), Ok(-30.0));
    assert!(b.det().is_err());
    assert_eq!(a.trace(), Ok(14));
}

#[test]
fn zero_one_diag_test() {
    let a = Matrix::from(vec![vec![0, 0, 0], vec![0, 0, 0]]).unwrap();
    let b = Matrix::from(vec![vec![1, 0], vec![0, 1]]).unwrap();

    assert_eq!(Matrix::<i32>::zero(2, 3), a);
    assert_eq!(Matrix::<i32>::identity(2), b);
    assert_eq!(Matrix::diagonal_matrix(vec![1, 1]), b);
}

#[test]
fn echelon_test() {
    let m = Matrix::from(vec![vec![1.0, 2.0, 3.0], vec![1.0, 0.0, 1.0]]).unwrap();
    let a = Matrix::from(vec![vec![1.0, 2.0, 3.0], vec![0.0, -2.0, -2.0]]).unwrap();
    let b = Matrix::from(vec![vec![1.0, 0.0, 0.0], vec![1.0, -2.0, 0.0]]).unwrap();
    let c = Matrix::from(vec![vec![1.0, 2.0, 3.0], vec![0.0, 1.0, 1.0]]).unwrap();

    assert_eq!(m.row_echelon(), a);
    assert_eq!(m.column_echelon(), b);
    assert_eq!(m.reduced_row_echelon(), c);
}

#[test]
fn conversion_test() {
    let a = Matrix::from(vec![vec![1, 2, 3], vec![0, 1, 2]]).unwrap();
    let b = Matrix::from(vec![vec![1.0, 2.0, 3.0], vec![0.0, 1.0, 2.0]]).unwrap();

    use crate::MatrixInto;
    assert_eq!(b, a.clone().matrix_into());

    use crate::MatrixFrom;
    let c = Matrix::<f64>::matrix_from(a);
    assert_eq!(c, b);
}

#[test]
fn inverse_test() {
    let a = Matrix::from(vec![vec![1.0, 2.0], vec![1.0, 2.0]]).unwrap();
    let b = Matrix::from(vec![
        vec![1.0, 2.0, 3.0],
        vec![0.0, 1.0, 4.0],
        vec![5.0, 6.0, 0.0],
    ])
    .unwrap();
    let c = Matrix::from(vec![
        vec![-24.0, 18.0, 5.0],
        vec![20.0, -15.0, -4.0],
        vec![-5.0, 4.0, 1.0],
    ])
    .unwrap();

    println!("{:?}", a.inverse());
    assert!(a.inverse().is_err());
    assert_eq!(b.inverse(), Ok(c));
}
