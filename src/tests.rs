#[cfg(test)]
use crate::Matrix;
#[test]
fn mul_test() {
    let a = Matrix::from(vec![vec![1, 2, 4], vec![3, 4, 9]]).unwrap();
    let b = Matrix::from(vec![vec![1, 2], vec![2, 3], vec![5, 1]]).unwrap();
    let c = Matrix::from(vec![vec![25, 12], vec![56, 27]]).unwrap();
    assert_eq!(a * b, c);
}

#[test]
fn add_sub_test() {
    let a = Matrix::from(vec![vec![1, 2, 3], vec![0, 1, 2]]).unwrap();
    let b = Matrix::from(vec![vec![0, 0, 1], vec![2, 1, 3]]).unwrap();
    let c = Matrix::from(vec![vec![1, 2, 4], vec![2, 2, 5]]).unwrap();
    let d = Matrix::from(vec![vec![1, 2, 2], vec![-2, 0, -1]]).unwrap();
    assert_eq!(a.clone() + b.clone(), c);
    assert_eq!(a - b, d);
}

#[test]
fn det_test() {
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
}

#[test]
fn zero_one_test() {
    let a = Matrix::from(vec![vec![0, 0, 0], vec![0, 0, 0]]).unwrap();
    let b = Matrix::from(vec![vec![1, 0], vec![0, 1]]).unwrap();
    assert_eq!(Matrix::<i32>::zero(2, 3), a);
    assert_eq!(Matrix::<i32>::identity(2), b);
}
