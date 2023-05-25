use std::{
    fmt::{self, Debug, Display, Formatter},
    ops::{Add, Mul},
    result::Result,
};

#[derive(PartialEq, Debug)]
pub struct Matrix<T: Mul + Add> {
    entries: Vec<Vec<T>>,
}

impl<T: Mul + Add> Matrix<T> {
    pub fn from(entries: Vec<Vec<T>>) -> Result<Matrix<T>, &'static str> {
        let mut equal_rows = true;
        let row_len = entries[0].len();
        for row in &entries {
            if row_len != row.len() {
                equal_rows = false;
                break;
            }
        }
        if equal_rows {
            Ok(Matrix { entries })
        } else {
            Err("Unequal rows.")
        }
    }

    pub fn height(&self) -> usize {
        self.entries.len()
    }

    pub fn width(&self) -> usize {
        self.entries[0].len()
    }

    pub fn transpose(&self) -> Matrix<T>
    where
        T: Copy,
    {
        let mut out = Vec::new();
        for i in 0..self.width() {
            let mut column = Vec::new();
            for row in &self.entries {
                column.push(row[i]);
            }
            out.push(column)
        }
        Matrix { entries: out }
    }

    pub fn rows(&self) -> Vec<Vec<T>>
    where
        T: Copy,
    {
        self.entries.clone()
    }

    pub fn columns(&self) -> Vec<Vec<T>>
    where
        T: Copy,
    {
        self.transpose().entries
    }
}

impl<T: Debug + Mul + Add> Display for Matrix<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{:?}", self.entries)
    }
}

impl<T: Mul<Output = T> + Add<Output = T> + Copy> Mul for Matrix<T> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let width = self.width();
        if width != other.height() {
            panic!("Row length of first matrix must be same as column length of second matrix.");
        } else {
            let mut out = Vec::new();
            for row in self.rows() {
                let mut new_row = Vec::new();
                for col in other.columns() {
                    let mut prod = row[0] * col[0];
                    for i in 1..width {
                        prod = prod + (row[i] * col[i]);
                    }
                    new_row.push(prod)
                }
                out.push(new_row);
            }
            Matrix { entries: out }
        }
    }
}

impl<T: Add<Output = T> + Mul + Copy> Add for Matrix<T> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        if self.height() == other.height() && self.width() == other.width() {
            let mut out = self.entries.clone();
            for (i, row) in self.rows().iter().enumerate() {
                for (j, entry) in other.rows()[i].iter().enumerate() {
                    out[i][j] = row[j] + *entry;
                }
            }
            Matrix { entries: out }
        } else {
            panic!("Both matrices must be of same dimensions.");
        }
    }
}
