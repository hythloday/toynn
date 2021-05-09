use ndarray::{azip, arr2, Array2};
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

// derivative of the sigmoid function
pub fn d_sigmoid(z: &f64) -> f64 {
    use std::f64::consts::E;
    E.powf(-z) / (1. + E.powf(-z)).powf(2.)
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

    /*
        loss = 2 * (y-output) * d_sig(output)
        weights1 += self.input.T . ((loss . self.weights2.T) * d_sig(layer1))
        weights2 += = self.layer1.T . loss
    */
    pub fn bp(self: &mut Self) {
        let mut loss = Array2::zeros(self.y.dim()); azip!((l in &mut loss, y in &self.y, o in &self.output) *l = 2. * (y - o) * d_sigmoid(o));
        azip!((w in &mut self.weights2, d in &self.layer1.t().dot(&loss)) *w += d);
        let mut chain = Array2::zeros(self.layer1.dim()); azip!((c in &mut chain, a in &loss.dot(&self.weights2.t()), b in &self.layer1) *c = a * d_sigmoid(b));
        azip!((w in &mut self.weights1, d in &self.input.t().dot(&chain)) *w += d);
    }
}

fn main() {
    let input = arr2(&[
        [0., 0., 1.],
        [0., 1., 1.],
        [1., 0., 1.],
        [1., 1., 1.]
    ]);
    let y = arr2(&[[0.], [1.], [1.], [0.]]);
    let mut nn = NeuralNetwork::new(&input, &y);
    for _ in 0..1500 {
        nn.ff();
        nn.bp();
    }
    println!("{:?}", nn.output);
}
