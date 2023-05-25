pub mod matrix;

#[cfg(test)]
mod tests {
    use super::*;
    use matrix::Matrix;
    #[test]
    fn mul_test() {
        let a = Matrix::from(vec![vec![1, 2, 4], vec![3, 4, 9]]).unwrap();
        let b = Matrix::from(vec![vec![1, 2], vec![2, 3], vec![5, 1]]).unwrap();
        let c = Matrix::from(vec![vec![25, 12], vec![56, 27]]).unwrap();
        assert_eq!(a * b, c);
    }

    #[test]
    fn add_test() {
        let a = Matrix::from(vec![vec![1, 2, 3], vec![0, 1, 2]]).unwrap();
        let b = Matrix::from(vec![vec![0, 0, 1], vec![2, 1, 3]]).unwrap();
        let c = Matrix::from(vec![vec![1, 2, 4], vec![2, 2, 5]]).unwrap();
        assert_eq!(a + b, c);
    }

    #[test]
    fn det_test() {
        let a = Matrix::from(vec![vec![1, 2, 0], vec![0, 3, 5], vec![0, 0, 10]]).unwrap();
        let b = Matrix::from(vec![vec![1, 2, 0], vec![0, 3, 5]]).unwrap();
        assert_eq!(a.det(), Ok(30));
        assert!(b.det().is_err());
    }
}
