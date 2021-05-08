use ndarray::{arr2, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

#[derive(Debug)]
struct NeuralNetwork {
    input: Array2<f64>,
    weights1: Array2<f64>,
    layer1: Array2<f64>,
    weights2: Array2<f64>,
    y: Array2<f64>,
    output: Array2<f64>
}

// sigmoid function.
pub fn sigmoid(z: &f64) -> f64 {
    use std::f64::consts::E;
    1. / (1. + E.powf(-z))
}

impl NeuralNetwork {
    pub fn new(x: &Array2<f64>, y: &Array2<f64>) -> NeuralNetwork {
        NeuralNetwork{
            input: x.clone(),
            weights1: Array2::random((x.dim().1, 4), Uniform::new(0., 1.)),
            layer1: Array2::zeros(y.dim()),
            weights2: Array2::random((4, 1), Uniform::new(0., 1.)),
            y: y.clone(),
            output: Array2::zeros(y.dim())
        }
    }

    pub fn ff(self: &mut Self) {
        self.layer1 = self.input.dot(&self.weights1).map(sigmoid);
        self.output = self.layer1.dot(&self.weights2).map(sigmoid);
    }
}

fn main() {
    let input = arr2(&[
        [0., 0., 1.],
        [0., 1., 1.],
        [1., 0., 1.],
        [1., 1., 1.]
    ]).t().into_owned();
    let y = arr2(&[[1.], [0.], [0.], [1.]]);
    let mut nn = NeuralNetwork::new(&input, &y);
    nn.ff();
    println!("{:?}", nn.layer1);

}
