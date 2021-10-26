use pyo3::prelude::*;
use pyo3::exceptions::*;
use numpy::*;


#[pymodule]
fn vitrix(py: Python, m: &PyModule) -> PyResult<()> {

    register_dynamics(py, m)?;
    
    py.run("\
import sys
sys.modules['vitrix.dynamics'] = dynamics
    ", None, Some(m.dict()))?;
    Ok(())
}

fn register_dynamics(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let child_module = PyModule::new(py, "dynamics")?;
    child_module.add_function(
        wrap_pyfunction!(nonaffine_local_strain_py, child_module)?
    )?;
    child_module.add_function(
        wrap_pyfunction!(affine_local_strain_py, child_module)?
    )?;
    parent_module.add_submodule(child_module)?;
    Ok(())
}

#[pyfunction(name="nonaffine_local_strain")]
fn nonaffine_local_strain_py(
        _py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        ) -> PyResult<f64> {
    let x = x.as_array();
    let y = y.as_array();
    dynamics::nonaffine_local_strain(x, y)
        .map_err(|e| PyArithmeticError::new_err(format!("{}", e)))
}


#[pyfunction(name="affine_local_strain")]
fn affine_local_strain_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        ) -> PyResult<&'py PyArray2<f64>> {
    let x = x.as_array();
    let y = y.as_array();
    match dynamics::affine_local_strain(x, y) {
        Ok(j) => Ok(j.into_pyarray(py)),
        Err(e) => Err(PyArithmeticError::new_err(format!("{}", e)))
    }
    
}

pub mod dynamics {
    use num::{Float, Zero};
    use ndarray::prelude::*;
    use ndarray_linalg::*;
    use ndarray_linalg::error::LinalgError;

    /// Calculates the affine local strain of particles given the local
    /// configurations at two times
    /// 
    #[inline(always)]
    pub fn affine_local_strain<T: Float + Scalar + Lapack>(
        initial_vectors: ArrayView2<T>,
        final_vectors: ArrayView2<T>)
    -> Result<Array2<T>, LinalgError> {
        let v = initial_vectors.t().dot(&initial_vectors);
        let w = initial_vectors.t().dot(&final_vectors);
        Ok(v.inv()?.dot(&w))
    }


    /// Calculates the nonaffine and affine local strain of particles given the
    /// local configurations at two times
    ///
    /// # Arguments
    /// 
    /// * `initial_vectors` - A string slice that holds the name of the person
    /// * `final_vectors` - A string slice that holds the name of the person
    /// 
    /// # Returns
    /// 
    /// * $D^2_{min}$
    /// * $J$
    ///
    #[inline(always)]
    pub fn nonaffine_and_affine_local_strain<T: Float + Scalar + Lapack>(
        initial_vectors: ArrayView2<T>,
        final_vectors: ArrayView2<T>)
    -> Result<(T, Array2<T>), LinalgError> {
        
        let j = affine_local_strain(initial_vectors, final_vectors)?;
        let non_affine = initial_vectors.dot(&j) - initial_vectors;
        let d2min = non_affine
            .iter()
            .fold(Zero::zero(), |sum: T, x| sum + (*x)*(*x));

        Ok((d2min, j))
    }


    /// Calculates the nonaffine local strain of particles given the
    /// local configurations at two times
    ///
    /// # Arguments
    /// 
    /// * `initial_vectors` - A string slice that holds the name of the person
    /// * `final_vectors` - A string slice that holds the name of the person
    /// 
    /// # Returns
    /// 
    /// * $D^2_{min}$
    ///
    #[inline(always)]
    pub fn nonaffine_local_strain<T: Float + Scalar + Lapack>(
        initial_bonds: ArrayView2<T>,
        final_bonds: ArrayView2<T>)
    -> Result<T, LinalgError> {
        let (d2min, _) = nonaffine_and_affine_local_strain::<T>(initial_bonds, final_bonds)?;
        Ok(d2min)
    }
}

mod statics {
    
}

mod ml {
    
}