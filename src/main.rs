use ndarray::{arr1, Array1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

#[derive(Debug)]
struct NeuralNetwork {
    input: Array1<f64>,
    weights1: Array1<f64>,
    weights2: Array1<f64>,
    y: Array1<f64>,
    output: Array1<f64>
}

impl NeuralNetwork {
    pub fn new(x: &Array1<f64>, y: &Array1<f64>) -> NeuralNetwork {
        NeuralNetwork{
            input: x.clone(),
            weights1: Array1::random(4, Uniform::new(0., 1.)),
            weights2: Array1::random(4, Uniform::new(0., 1.)),
            y: y.clone(),
            output: Array1::zeros(y.len())

        }
    }
}

fn main() {
    println!("{:?}", NeuralNetwork::new(&arr1(&[1., 2., 3., 4.]), &arr1(&[1., 0., 0., 1.])));
}
